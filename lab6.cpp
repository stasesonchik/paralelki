#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <boost/program_options.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Options {
    int N;
    double tol;
    long maxIter;
    std::string device;
};

Options parseOptions(int argc, char** argv, bool& run_single_solver) {
    namespace po = boost::program_options;
    Options opt;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help")
        ("run-single", "run single simulation with parameters")
        ("size", po::value<int>(&opt.N)->default_value(128))
        ("tol", po::value<double>(&opt.tol)->default_value(1e-6))
        ("max-iter", po::value<long>(&opt.maxIter)->default_value(1000000))
        ("device", po::value<std::string>(&opt.device)->default_value("gpu"), "device to use: cpu or gpu");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }
    run_single_solver = vm.count("run-single");
    po::notify(vm);
    return opt;
}

void initBoundary(double* A, int N) {
    const double tl = 10.0, tr = 20.0, br = 30.0, bl = 20.0;

#pragma acc parallel loop
    for (int i = 0; i < N; ++i) {
        A[i] = tl + (tr - tl) * i / (N - 1);
        A[(N - 1) * N + i] = bl + (br - bl) * i / (N - 1);
    }

#pragma acc parallel loop
    for (int j = 0; j < N; ++j) {
        A[j * N] = tl + (bl - tl) * j / (N - 1);
        A[j * N + (N - 1)] = tr + (br - tr) * j / (N - 1);
    }
}

void initInterior(double* A, int N) {
#pragma omp parallel for collapse(2)
    for (int j = 1; j < N - 1; ++j)
        for (int i = 1; i < N - 1; ++i)
            A[j * N + i] = 0.0;
}

double simpleIterationCPU(double* A, double* Anew, int N) {
    double maxError = 0.0;

#pragma omp parallel for reduction(max:maxError)
    for (int j = 1; j < N - 1; ++j) {
        for (int i = 1; i < N - 1; ++i) {
            int idx = j * N + i;
            Anew[idx] = 0.25 * (A[idx - 1] + A[idx + 1] + A[idx - N] + A[idx + N]);
            double diff = std::fabs(Anew[idx] - A[idx]);
            if (diff > maxError) maxError = diff;
        }
    }

    return maxError;
}

double simpleIterationGPU(double* A, double* Anew, int N) {
    double maxError = 0.0;

#pragma acc parallel loop collapse(2) reduction(max:maxError)
    for (int j = 1; j < N - 1; ++j) {
        for (int i = 1; i < N - 1; ++i) {
            int idx = j * N + i;
            Anew[idx] = 0.25 * (A[idx - 1] + A[idx + 1] + A[idx - N] + A[idx + N]);
            double diff = std::fabs(Anew[idx] - A[idx]);
            if (diff > maxError) maxError = diff;
        }
    }
    return maxError;
}

void runSolverGPU(int N, double tol, long maxIter) {
    int NM = N * N;
    double* A = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));
    double* Anew = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));

#pragma acc data copy(A[0:NM]), create(Anew[0:NM])
    {
        initBoundary(A, N);
        initInterior(A, N);
#pragma acc parallel loop collapse(2)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                Anew[j * N + i] = A[j * N + i];

        long iter = 0;
        double maxErr = 0.0;
        auto t_start = std::chrono::high_resolution_clock::now();

        do {
            maxErr = simpleIterationGPU(A, Anew, N);
#pragma acc parallel loop collapse(2)
            for (int j = 1; j < N - 1; ++j)
                for (int i = 1; i < N - 1; ++i)
                    A[j * N + i] = Anew[j * N + i];
            ++iter;
        } while (maxErr > tol && iter < maxIter);

        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;

        std::cout << "Simple Iteration [OpenACC GPU] - Grid: " << N
                  << ", Iterations: " << iter
                  << ", Final Error: " << maxErr
                  << ", Time: " << elapsed.count() << " sec\n";

        if (N == 10 || N == 13) {
#pragma acc update self(A[0:NM])
            std::ofstream fout("simple_output_gpu.csv");
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    fout << A[j * N + i];
                    if (i < N - 1) fout << ",";
                }
                fout << "\n";
            }
            std::cout << "Matrix saved to simple_output_gpu.csv\n";
        }
    }

    free(A);
    free(Anew);
}

void runSolverCPU(int N, double tol, long maxIter) {
    int NM = N * N;
    double* A = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));
    double* Anew = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));

    initBoundary(A, N);
    initInterior(A, N);
    std::memcpy(Anew, A, NM * sizeof(double));

    long iter = 0;
    double maxErr = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();

    do {
        maxErr = simpleIterationCPU(A, Anew, N);
        std::swap(A, Anew);
        ++iter;
    } while (maxErr > tol && iter < maxIter);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    std::cout << "Grid: " << N
              << ", Iterations: " << iter
              << ", Final Error: " << maxErr
              << ", Time: " << elapsed.count() << " sec\n";

    if (N == 10 || N == 13) {
        std::ofstream fout("simple_output_cpu.csv");
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                fout << A[j * N + i];
                if (i < N - 1) fout << ",";
            }
            fout << "\n";
        }
        std::cout << "Matrix saved to simple_output_cpu.csv\n";
    }

    free(A);
    free(Anew);
}

void runBenchmarks(double tol, long maxIter) {
    std::cout << "===== GPU Benchmarks (OpenACC Simple Iteration) =====\n";
    for (int N : {128, 256, 512, 1024}) {
        runSolverGPU(N, tol, maxIter);
    }

    std::cout << "\n===== CPU Benchmarks (OpenMP Simple Iteration) =====\n";
    for (int N : {128, 256, 512, 1024}) {
        runSolverCPU(N, tol, maxIter);
    }
}

int main(int argc, char** argv) {
    bool run_single_solver = false;
    Options opt = parseOptions(argc, argv, run_single_solver);

    if (run_single_solver) {
        if (opt.device == "gpu")
            runSolverGPU(opt.N, opt.tol, opt.maxIter);
        else if (opt.device == "cpu")
            runSolverCPU(opt.N, opt.tol, opt.maxIter);
        else {
            std::cerr << "Invalid device: " << opt.device << ". Use --device=gpu or --device=cpu\n";
            return 1;
        }
    } else {
        runBenchmarks(opt.tol, opt.maxIter);
    }

    return 0;
}
