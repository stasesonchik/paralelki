#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

namespace opt = boost::program_options;

// Линейная интерполяция
double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

// Сохранение матрицы в файл
void saveMatrixToFile(const double* matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи." << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(10) << std::fixed << std::setprecision(4)
                       << matrix[i * size + j];
        }
        outputFile << '\n';
    }
}

// Зануление внутренней области
void zeroInternalArea(std::vector<double>& matrix, size_t size) {
    for (size_t i = 1; i < size - 1; ++i) {
        for (size_t j = 1; j < size - 1; ++j) {
            matrix[i * size + j] = 0.0;
        }
    }
}

// Инициализация краёв и углов
void initializeMatrix(std::vector<double>& matrix, int size) {
    matrix[0] = 10.0;
    matrix[size - 1] = 20.0;
    matrix[(size - 1) * size] = 20.0;
    matrix[(size - 1) * size + (size - 1)] = 30.0;

    for (int i = 1; i < size - 1; i++) {
        matrix[0 * size + i] = linearInterpolation(i, 0.0, matrix[0], size - 1, matrix[size - 1]);
        matrix[i * size + 0] = linearInterpolation(i, 0.0, matrix[0], size - 1, matrix[(size - 1) * size]);
        matrix[i * size + (size - 1)] = linearInterpolation(i, 0.0, matrix[size - 1], size - 1, matrix[(size - 1) * size + (size - 1)]);
        matrix[(size - 1) * size + i] = linearInterpolation(i, 0.0, matrix[(size - 1) * size], size - 1, matrix[(size - 1) * size + (size - 1)]);
    }
}

int main(int argc, char const* argv[]) {
    opt::options_description desc("Опции");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Точность")
        ("size", opt::value<int>()->default_value(1024), "Размер матрицы")
        ("iterations", opt::value<int>()->default_value(1000000), "Максимум итераций")
        ("help", "Помощь");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    int matrixSize = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int maxIterations = vm["iterations"].as<int>();

    std::vector<double> matrixA(matrixSize * matrixSize);
    std::vector<double> matrixB(matrixSize * matrixSize);

    initializeMatrix(matrixA, matrixSize);
    initializeMatrix(matrixB, matrixSize);
    zeroInternalArea(matrixA, matrixSize);
    zeroInternalArea(matrixB, matrixSize);

    double* previousMatrix = matrixA.data();
    double* updatedMatrix = matrixB.data();

    double error = 1.0;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now();
    #pragma acc data copyin(previousMatrix[0:matrixSize*matrixSize], updatedMatrix[0:matrixSize*matrixSize]) copy(error)
    {
        while (iteration < maxIterations && error > accuracy) {
            #pragma acc parallel loop collapse(2) present(previousMatrix, updatedMatrix)
            for (int i = 1; i < matrixSize - 1; ++i) {
                for (int j = 1; j < matrixSize - 1; ++j) {
                    updatedMatrix[i * matrixSize + j] = 0.25 * (
                        previousMatrix[i * matrixSize + j + 1] +
                        previousMatrix[i * matrixSize + j - 1] +
                        previousMatrix[(i - 1) * matrixSize + j] +
                        previousMatrix[(i + 1) * matrixSize + j]);
                }
            }

            // Вычисляем ошибку
            error = 0.0;
            #pragma acc update device(error)
            #pragma acc parallel loop collapse(2) reduction(max:error) present(previousMatrix, updatedMatrix)
            for (int i = 1; i < matrixSize - 1; ++i) {
                for (int j = 1; j < matrixSize - 1; ++j) {
                    error = std::max(error, std::abs(updatedMatrix[i * matrixSize + j] - previousMatrix[i * matrixSize + j]));
                }
            }
            #pragma acc update self(error)

            std::swap(previousMatrix, updatedMatrix);
            iteration++;
        }

        #pragma acc update self(previousMatrix[0:matrixSize*matrixSize])
    }
    auto end = std::chrono::high_resolution_clock::now();

    double timeInSeconds = std::chrono::duration<double>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Время: " << timeInSeconds << " секунд\n";
    std::cout << "Финальная ошибка: " << error << "\n";
    std::cout << "Количество итераций: " << iteration << "\n";

    if (matrixSize == 10 || matrixSize == 13) {
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                std::cout << previousMatrix[i * matrixSize + j] << ' ';
            }
            std::cout << '\n';
        }
    }

    saveMatrixToFile(previousMatrix, matrixSize, "matrix.txt");
    return 0;
}
