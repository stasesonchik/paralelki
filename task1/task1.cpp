#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

struct VT_pair {
    std::vector<double> res;
    double time;
};

void print_arrays(double *a, double *b, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            std::cout << a[i * m + j] << "  ";
        std::cout << std::endl;
    }

    std::cout << "\n";

    for (int i = 0; i < n; i++)
        std::cout << b[i] << "  ";
    std::cout << std::endl;
}

// Параллельная инициализация
void init_arrays(std::vector<double>& matrix, std::vector<double>& vector, int n, int threads)
{
    int rows_per_thread = n / threads;
    std::vector<std::thread> thread_pool;

    for (int t = 0; t < threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = (t == threads - 1) ? n : row_start + rows_per_thread;

        thread_pool.emplace_back([&, row_start, row_end]() {
            for (int i = row_start; i < row_end; ++i) {
                for (int j = 0; j < n; ++j) {
                    matrix[i * n + j] = i + j;
                }
            }
        });
    }

    for (auto& thread : thread_pool)
        thread.join();

    for (int i = 0; i < n; ++i)
        vector[i] = i;
}

VT_pair run_serial(int n)
{
    std::vector<double> matrix(n * n);
    std::vector<double> vector(n);
    std::vector<double> res(n, 0.0);

    auto start = std::chrono::steady_clock::now();

    // Инициализация
    init_arrays(matrix, vector, n, 1);

    // Перемножение
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i] += matrix[i * n + j] * vector[j];
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time = end - start;

    return {res, time.count()};
}

void matrix_vector_product_row(const double* a, const double* b, double* c, int m, int n, int row_start, int row_end)
{
    for (int i = row_start; i < row_end; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

// Параллельный запуск + параллельная инициализация
VT_pair run_parallel(int n, int THREADS)
{
    std::vector<double> matrix(n * n);
    std::vector<double> vector(n);
    std::vector<double> res(n, 0.0);
    int rows_per_thread = n / THREADS;
    std::vector<std::thread> threads;

    auto start = std::chrono::steady_clock::now();

    init_arrays(matrix, vector, n, THREADS);

    for (int i = 0; i < THREADS; i++) {
        int row_start = i * rows_per_thread;
        int row_end = (i == THREADS - 1) ? n : row_start + rows_per_thread;

        threads.emplace_back(matrix_vector_product_row,
                             matrix.data(), vector.data(), res.data(), n, n,
                             row_start, row_end);
    }

    for (auto& t : threads)
        t.join();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time = end - start;

    return {res, time.count()};
}

void Matrix_multipletion(int N)
{
    double serial_result_time = 0.0;
    std::vector<int> thread_counts = {1, 2, 4, 7, 8, 16, 20, 40};

    std::vector<double> matrix(N * N);
    std::vector<double> vector(N);
    init_arrays(matrix, vector, N, 8);  // Инициализация для serial версии

    std::cout << "N - " << N << "\n";

    VT_pair serial_result_N1 = run_serial(N);
    serial_result_time = serial_result_N1.time;

    for (int threads : thread_counts)
    {
        VT_pair parallel_result = run_parallel(N, threads);

        std::cout << "threads - " << threads << "\n";
        std::cout << "T1 - " << serial_result_time << "\n";
        std::cout << "T2 - " << parallel_result.time << "\n";
        std::cout << "Speedup = " << serial_result_time / parallel_result.time << "\n__________________________\n" << std::endl;
    }
}

void Check_parametrs()
{
    int N = 20000;
    int threads = 20;

    std::vector<double> matrix(N * N);
    std::vector<double> vector(N);
    init_arrays(matrix, vector, N, threads);

    VT_pair serial_result = run_serial( N);
    VT_pair parallel_result = run_parallel(N, threads);

    std::cout << "threads - " << threads << "\n";
    std::cout << "T1 - " << serial_result.time << "\n";
    std::cout << "T2 - " << parallel_result.time << "\n";
}

int main(int argc, char** argv)
{
    Matrix_multipletion(20000);
    Matrix_multipletion(40000);
    //Check_parametrs();
    return 0;
}
