#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>
#include <map>

// Определения
#define TAU 0.01
#define EPS 0.0001

// Функция для вычисления евклидовой нормы
double euclid_norm(const std::vector<double>& vec, int N) {
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += pow(vec[i], 2.0);
    }
    return sqrt(norm);
}

// Простой итерационный метод для решения Ax = b
std::vector<double> simple_iteration(std::vector<std::vector<double>>& A,
                                    std::vector<double>& x,
                                    std::vector<double>& b,
                                    int matrix_size,
                                    int num_threads,
                                    const std::string& schedule_type)
{
    std::vector<double> Ax(matrix_size, 0.0);
    double Ax_norm = EPS + 1;  // Для начала больше, чем EPS

    #pragma omp parallel num_threads(num_threads)
    {
        while (Ax_norm > EPS) {
            // Вычисление Ax для каждого элемента
            if (schedule_type == "static") {
                #pragma omp for schedule(static)
                for (int i = 0; i < matrix_size; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < matrix_size; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            }
            else if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < matrix_size; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < matrix_size; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            }
            else if (schedule_type == "guided") {
                #pragma omp for schedule(guided)
                for (int i = 0; i < matrix_size; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < matrix_size; j++) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            }

            // Обновление x
            #pragma omp for
            for (int i = 0; i < matrix_size; i++) {
                x[i] = x[i] - TAU * (Ax[i] - b[i]);
            }

            // Вычисление нормы вектора Ax
            #pragma omp single
            Ax_norm = euclid_norm(Ax, matrix_size);
        }
    }

    return x;
}

double run(int matrix_size, int num_threads, const std::string& schedule_type) {
    std::vector<std::vector<double>> A(matrix_size, std::vector<double>(matrix_size, 1.0));
    std::vector<double> x(matrix_size, 0.0);  // Начальные значения x
    std::vector<double> b(matrix_size, matrix_size + 1);  // Вектор b (все элементы равны N + 1)

    for (int i = 0; i < matrix_size; i++) {
        A[i][i] = 2.0;  // Главная диагональ равна 2.0
    }

    const auto start = std::chrono::steady_clock::now();
    x = simple_iteration(A, x, b, matrix_size, num_threads, schedule_type);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    return elapsed_seconds.count();
}

int main(int argc, char* argv[]) {
    
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <schedule_type>\n";
        std::cout << "Valid options for <schedule_type>: static, dynamic, guided\n";
        return 1;
    }

    // Чтение параметра из командной строки
    std::string schedule_type = argv[1];

    // Проверка корректности введённого типа планирования
    if (schedule_type != "static" && schedule_type != "dynamic" && schedule_type != "guided") {
        std::cout << "Invalid schedule type. Valid options are: static, dynamic, guided.\n";
        return 1;
    }

    int matrix_size = 40000;  // Размерность задачи
    double tserial, tparallel;

    // Замер времени для последовательного исполнения
    tserial = run(matrix_size, 1, schedule_type);
    std::cout << "Elapsed time (serial): " << tserial << " seconds\n";

    // Замер времени для параллельного исполнения с разным количеством потоков
    std::vector<int> thread_counts = {2, 4, 8, 16, 20, 40};  // Количество потоков

    for (int threads : thread_counts) {
        tparallel = run(matrix_size, threads, schedule_type);
        std::cout << " Threads: " << threads
                  << " | Time: " << tparallel
                  << " | Speedup: " << tserial / tparallel << "\n";
        std::cout << "------------------------------------\n";
    }

    return 0;
}
