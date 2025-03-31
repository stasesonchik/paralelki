#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

struct VT_pair {
    std::vector<double> res;
    double time;
};

void init_arrays(std::vector<double>& matrix, std::vector<double>& vector, int n) {
    std::vector<std::thread> threads;
    int THREADS = std::thread::hardware_concurrency();
    int chunk_size = n / THREADS;

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            for (int j = 0; j < n; j++)
                matrix[i * n + j] = i + j;
            vector[i] = i;
        }
    };

    for (int t = 0; t < THREADS; t++) {
        int start = t * chunk_size;
        int end = (t == THREADS - 1) ? n : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) {
        th.join();
    }
}

VT_pair run_serial(const std::vector<double>& matrix, const std::vector<double>& vector, int n) {
    std::vector<double> res(n, 0.0);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i] += matrix[i * n + j] * vector[j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    return {res, time};
}

VT_pair run_parallel_threads(const std::vector<double>& matrix, const std::vector<double>& vector, int n, int THREADS) {
    std::vector<double> res(n, 0.0);
    std::vector<std::thread> threads;
    auto start = std::chrono::high_resolution_clock::now();

    auto worker = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < n; j++) {
                res[i] += matrix[i * n + j] * vector[j];
            }
        }
    };

    int chunk_size = n / THREADS;
    for (int t = 0; t < THREADS; t++) {
        int start_row = t * chunk_size;
        int end_row = (t == THREADS - 1) ? n : start_row + chunk_size;
        threads.emplace_back(worker, start_row, end_row);
    }

    for (auto& th : threads) {
        th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    return {res, time};
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
        VT_pair parallel_result = run_parallel_threads(matrix, vector, N, threads);
        std::cout << "[std::thread] Threads: " << threads 
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
