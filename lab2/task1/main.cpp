#include <iostream>
#include <vector>
#include <omp.h>

struct VT_pair {
    std::vector<double> res;
    double time;
};

void init_arrays(std::vector<double>& matrix, std::vector<double>& vector, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            matrix[i * n + j] = i + j;
        vector[i] = i;
    }
}

VT_pair run_serial(const std::vector<double>& matrix, const std::vector<double>& vector, int n) {
    std::vector<double> res(n, 0.0);
    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i] += matrix[i * n + j] * vector[j];
        }
    }

    double end = omp_get_wtime();
    return {res, end - start};
}

VT_pair run_parallel_omp(const std::vector<double>& matrix, const std::vector<double>& vector, int n, int THREADS) {
    std::vector<double> res(n, 0.0);
    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * vector[j];
        }
        res[i] = sum;
    }

    double end = omp_get_wtime();
    return {res, end - start};
}

void Matrix_multiplication(int N) {
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 40};
    std::vector<double> matrix(N * N);
    std::vector<double> vector(N);
    init_arrays(matrix, vector, N);

    std::cout << "N = " << N << "\n";

    VT_pair serial_result = run_serial(matrix, vector, N);
    double serial_time = serial_result.time;
    std::cout << "Serial time: " << serial_time << " sec\n";

    for (int threads : thread_counts) {
        VT_pair parallel_result = run_parallel_omp(matrix, vector, N, threads);
        std::cout << "[OpenMP] Threads: " << threads 
                  << " | Time: " << parallel_result.time
                  << " | Speedup: " << serial_time / parallel_result.time << "\n";
        std::cout << "------------------------------------\n";
    }
}

int main() {
    Matrix_multiplication(20000);
    Matrix_multiplication(40000);
    return 0;
}
