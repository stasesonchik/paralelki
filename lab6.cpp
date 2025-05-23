#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <boost/program_options.hpp>

// =============================
// ПАРАМЕТРЫ
// =============================
struct Options {
    int N;
    double tol;
    long maxIter;
};

Options parseOptions(int argc, char** argv) {
    namespace po = boost::program_options;
    Options opt;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help")
        ("size", po::value<int>(&opt.N)->default_value(128))
        ("tol", po::value<double>(&opt.tol)->default_value(1e-6))
        ("max-iter", po::value<long>(&opt.maxIter)->default_value(1000000));

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }
    po::notify(vm);
    return opt;
}

// =============================
// ИНИЦИАЛИЗАЦИЯ
// =============================
void initInterior(double* __restrict A, int N) {
    for (int j = 1; j < N - 1; ++j) {
        double* row = A + j * N;
        for (int i = 1; i < N - 1; ++i) {
            row[i] = 0.0;
        }
    }
}

void initBoundary(double* __restrict A, int N) {
    const double tl = 10.0;
    const double tr = 20.0;
    const double br = 30.0;
    const double bl = 20.0;

    for (int i = 0; i < N; ++i) {
        A[i] = tl + (tr - tl) * static_cast<double>(i) / (N - 1);
    }
    for (int j = 0; j < N; ++j) {
        A[j * N + (N - 1)] = tr + (br - tr) * static_cast<double>(j) / (N - 1);
    }
    for (int i = 0; i < N; ++i) {
        A[(N - 1) * N + i] = bl + (br - bl) * static_cast<double>(i) / (N - 1);
    }
    for (int j = 0; j < N; ++j) {
        A[j * N] = tl + (bl - tl) * static_cast<double>(j) / (N - 1);
    }
}

// =============================
// ОСНОВНАЯ ИТЕРАЦИЯ
// =============================
double jacobiIteration(const double* __restrict A, double* __restrict Anew, int N) {
    int NM = N * N;
    double maxError = 0.0;

    #pragma acc parallel loop collapse(2) present(A[0:NM], Anew[0:NM]) reduction(max:maxError)
    for (int j = 1; j < N - 1; ++j) {
        for (int i = 1; i < N - 1; ++i) {
            int idx = j * N + i;
            double v = 0.25 * (A[idx-1] + A[idx+1] + A[idx+N] + A[idx-N]);
            Anew[idx] = v;
            double diff = std::fabs(v - A[idx]);
            if (diff > maxError) maxError = diff;
        }
    }
    return maxError;
}

void printSummary(long iter, double maxError) {
    std::cout << "Iterations: " << iter << ", Max Error: " << maxError << "\n";
}

int main(int argc, char** argv) {
    auto opt = parseOptions(argc, argv);
    int N  = opt.N;
    int NM = N * N;

    double* A    = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));
    double* Anew = static_cast<double*>(aligned_alloc(64, NM * sizeof(double)));

    initBoundary(A,    N);
    initBoundary(Anew, N);
    initInterior(A,    N);
    initInterior(Anew, N);

    long iter = 0;
    double maxErr = 0.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    #pragma acc data copy(A[0:NM]) copy(Anew[0:NM])
    {
        do {
            maxErr = jacobiIteration(A, Anew, N);
            std::swap(A, Anew);
            ++iter;
            if (N <= 20) std::cout << "Iteration " << iter << ": maxError = " << maxErr << std::endl;
        } while (maxErr > opt.tol && iter < opt.maxIter);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    printSummary(iter, maxErr);
    std::cout << "Elapsed time: " << elapsed.count() << " sec\n";

    if (N == 10 || N == 13) {
        std::ofstream fout("matrix_output.csv");
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                fout << A[j * N + i];
                if (i < N - 1) fout << ",";
            }
            fout << "\n";
        }
        std::cout << "Matrix saved to matrix_output.csv\n";
    }

    free(A);
    free(Anew);
    return 0;
}
