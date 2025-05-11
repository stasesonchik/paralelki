#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>

struct Result {
    double integral;
    double time;
};

double func(double x) {
    return cos(x);
}

Result integrate_serial(int nsteps, double a, double b) {
    double sum = 0;
    double h = (b - a) / nsteps;

    double start = omp_get_wtime();
    for (int i = 0; i < nsteps; i++)
        sum += func(a + (h * i) - (h / 2));
    double end = omp_get_wtime();

    return {sum * h, end - start};
}

Result integrate_parallel(int nsteps, double a, double b, int THREADS) {
    double sum = 0;
    double h = (b - a) / nsteps;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = nsteps / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? nsteps : lb + items_per_thread;
        double local_sum = 0;

        for (int i = lb; i < ub; i++)
            local_sum += func(a + (h * i) - (h / 2));

        #pragma omp atomic
        sum += local_sum;
    }

    double end = omp_get_wtime();
    return {sum * h, end - start};
}

void run_integration(int nsteps, double a, double b) {
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 40};

    // Serial integration
    Result serial_result = integrate_serial(nsteps, a, b);
    double serial_time = serial_result.time;
    std::cout << "Serial result: " << serial_result.integral << ", Time: " << serial_time << " sec\n";

    // Parallel integration with different thread counts
    for (int threads : thread_counts) {
        Result parallel_result = integrate_parallel(nsteps, a, b, threads);
        double speedup = serial_time / parallel_result.time;

        std::cout << "Threads: " << threads << "\n";
        std::cout << "Parallel result: " << parallel_result.integral << ", Time: " << parallel_result.time << " sec\n";
        std::cout << "Speedup: " << speedup << "x\n";
        std::cout << "-------------------------------------\n";
    }
}

int main() 
{
    int nsteps = 40000000;
    double a = -4.0, b = 4.0;


    run_integration(nsteps, a, b);

    return 0;
}
