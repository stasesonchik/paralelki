#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>

#define TAU 0.01
#define EPS 0.0001

double euclid_norm(const std::vector<double>& vec, int N) {
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += vec[i] * vec[i];
    }
    return sqrt(norm);
}

std::vector<double> simple_iteration(const std::vector<std::vector<double>>& A,
                                     std::vector<double> x,  // копия, чтобы исходный не менялся
                                     const std::vector<double>& b,
                                     int matrix_size,
                                     int num_threads,
                                     const std::string& schedule_type)
{
    std::vector<double> Ax(matrix_size, 0.0);
    double Ax_norm = EPS + 1;

    #pragma omp parallel num_threads(num_threads)
    {
        while (Ax_norm > EPS) {

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

            #pragma omp for
            for (int i = 0; i < matrix_size; i++) {
                x[i] = x[i] - TAU * (Ax[i] - b[i]);
            }

            #pragma omp single
            Ax_norm = euclid_norm(Ax, matrix_size);
        }
    }

    return x;
}

double run(const std::vector<std::vector<double>>& A,
           const std::vector<double>& b,
           int matrix_size,
           int num_threads,
           const std::string& schedule_type)
{
    std::vector<double> x(matrix_size, 0.0);  // локальная копия x

    auto start = std::chrono::steady_clock::now();
    x = simple_iteration(A, x, b, matrix_size, num_threads, schedule_type);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <schedule_type>\n";
        std::cout << "Valid options: static, dynamic, guided\n";
        return 1;
    }

    std::string schedule_type = argv[1];
    if (schedule_type != "static" && schedule_type != "dynamic" && schedule_type != "guided") {
        std::cerr << "Invalid schedule type. Valid options: static, dynamic, guided\n";
        return 1;
    }

    int matrix_size = 40000;

    // Инициализация A и b один раз
    std::vector<std::vector<double>> A(matrix_size, std::vector<double>(matrix_size, 1.0));
    std::vector<double> b(matrix_size, matrix_size + 1);

    for (int i = 0; i < matrix_size; i++) {
        A[i][i] = 2.0;
    }

    double tserial = run(A, b, matrix_size, 1, schedule_type);
    std::cout << "Elapsed time (serial): " << tserial << " seconds\n";

    std::vector<int> thread_counts = {2, 4, 8, 16, 20, 40};

    for (int threads : thread_counts) {
        double tparallel = run(A, b, matrix_size, threads, schedule_type);
        std::cout << " Threads: " << threads
                  << " | Time: " << tparallel
                  << " | Speedup: " << tserial / tparallel << "\n";
        std::cout << "------------------------------------\n";
    }

    return 0;
}
