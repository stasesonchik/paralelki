#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cublas_v2.h>

namespace opt = boost::program_options;


double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}


void saveMatrixToFile(const std::vector<double>& matrix, int size, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Не могу открыть файл " << filename << " — возможно, он заблокирован или ты что-то напортачил.\n";
        return;
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            out << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        out << '\n';
    }
}


void zeroInternalArea(std::vector<double>& matrix, int size) {
    for (int i = 1; i < size - 1; ++i)
        for (int j = 1; j < size - 1; ++j)
            matrix[i * size + j] = 0.0;
}


void initializeMatrix(std::vector<double>& matrix, int size) {
    matrix[0] = 10.0;
    matrix[size - 1] = 20.0;
    matrix[(size - 1) * size] = 20.0;
    matrix[(size - 1) * size + (size - 1)] = 30.0;

    for (int i = 1; i < size - 1; ++i) {
        matrix[i] = linearInterpolation(i, 0, matrix[0], size - 1, matrix[size - 1]); 
        matrix[i * size] = linearInterpolation(i, 0, matrix[0], size - 1, matrix[(size - 1) * size]); 
        matrix[i * size + (size - 1)] = linearInterpolation(i, 0, matrix[size - 1], size - 1, matrix[(size - 1) * size + (size - 1)]); // Правая грань
        matrix[(size - 1) * size + i] = linearInterpolation(i, 0, matrix[(size - 1) * size], size - 1, matrix[(size - 1) * size + (size - 1)]); // Нижняя грань
    }
}

int main(int argc, char* argv[]) {

    opt::options_description desc("Опции");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Желаемая точность (чем меньше — тем дольше мучиться)")
        ("size", opt::value<int>()->default_value(1024), "Размер квадратной матрицы (1024 — не меньше, чтобы почувствовать мощь)")
        ("iterations", opt::value<int>()->default_value(1000000), "Максимальное число итераций (1000000 — чтоб успел посчитать)")
        ("help", "Вывести эту справку и убежать");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    const int matrixSize = vm["size"].as<int>();
    const double accuracy = vm["accuracy"].as<double>();
    const int maxIterations = vm["iterations"].as<int>();

  
    std::vector<double> matrixA(matrixSize * matrixSize);
    std::vector<double> matrixB(matrixSize * matrixSize);

    initializeMatrix(matrixA, matrixSize);
    initializeMatrix(matrixB, matrixSize);
    zeroInternalArea(matrixA, matrixSize);
    zeroInternalArea(matrixB, matrixSize);

    double* current = matrixB.data();
    double* previous = matrixA.data();


    std::vector<double> diff(matrixSize * matrixSize, 0.0);
    double* diffDevice = diff.data();


    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate — пиздец, не запустился\n";
        return EXIT_FAILURE;
    }

    double error = 1.0;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc enter data copyin(current[0:matrixSize*matrixSize], previous[0:matrixSize*matrixSize], diffDevice[0:matrixSize*matrixSize]) create(error)
    {
        while (iteration < maxIterations && error > accuracy) {
            #pragma acc parallel loop collapse(2) present(current, previous)
            for (int i = 1; i < matrixSize - 1; ++i) {
                for (int j = 1; j < matrixSize - 1; ++j) {
                    current[i * matrixSize + j] = 0.25 * (
                        previous[i * matrixSize + j + 1] +
                        previous[i * matrixSize + j - 1] +
                        previous[(i - 1) * matrixSize + j] +
                        previous[(i + 1) * matrixSize + j]
                    );
                }
            }


            if (iteration % 10000 == 0) {
                int maxIdx = 0;

                #pragma acc host_data use_device(current, previous, diffDevice)
                {
                    if (cublasDcopy(handle, matrixSize * matrixSize, current, 1, diffDevice, 1) != CUBLAS_STATUS_SUCCESS ||
                        cublasDaxpy(handle, matrixSize * matrixSize, new double(-1.0), previous, 1, diffDevice, 1) != CUBLAS_STATUS_SUCCESS ||
                        cublasIdamax(handle, matrixSize * matrixSize, diffDevice, 1, &maxIdx) != CUBLAS_STATUS_SUCCESS) 
                    {
                        std::cerr << "cublas ошибка — что-то пошло не так с вычислением ошибки\n";
                        cublasDestroy(handle);
                        return EXIT_FAILURE;
                    }
                }

                double host_error = 0.0;
                #pragma acc host_data use_device(diffDevice)
                {
                    if (maxIdx > 0 && maxIdx <= matrixSize * matrixSize) {
                        if (cublasGetVector(1, sizeof(double), diffDevice + (maxIdx - 1), 1, &host_error, 1) != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "cublasGetVector — идиотизм с индексом\n";
                            cublasDestroy(handle);
                            return EXIT_FAILURE;
                        }
                    }
                }

                error = std::abs(host_error);
            }

            std::swap(current, previous);
            ++iteration;
        }

        #pragma acc update self(current[0:matrixSize*matrixSize])
    }
    #pragma acc exit data delete(current[0:matrixSize*matrixSize], previous[0:matrixSize*matrixSize], diffDevice[0:matrixSize*matrixSize], error)

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Время работы: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " мс, ошибка: " << error << ", итераций: " << iteration << '\n';


    if (matrixSize == 10 || matrixSize == 13) {
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j)
                std::cout << previous[i * matrixSize + j] << ' ';
            std::cout << '\n';
        }
    }

    saveMatrixToFile(std::vector<double>(previous, previous + matrixSize * matrixSize), matrixSize, "matrix.txt");

    cublasDestroy(handle);
    return 0;
}
