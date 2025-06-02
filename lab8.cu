#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <boost/program_options.hpp>

namespace opt = boost::program_options;

__global__ void updateKernel(double* prev, double* curr, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < N - 1 && j < N - 1) {
        curr[i * N + j] = 0.25 * (
            prev[i * N + (j + 1)] +
            prev[i * N + (j - 1)] +
            prev[(i - 1) * N + j] +
            prev[(i + 1) * N + j]
        );
    }
}

__global__ void errorKernel(const double* curr, const double* prev, double* result, int N) {
    typedef cub::BlockReduce<double, 32, cub::BLOCK_REDUCE_RAKING, 16> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    double diff = 0.0;
    if (i < N - 1 && j < N - 1) {
        diff = fabs(curr[i * N + j] - prev[i * N + j]);
    }

    double block_max = BlockReduce(temp_storage).Reduce(diff, cub::Max());

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        result[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}

__global__ void finalMaxReduceKernel(const double* input, double* result, int N) {
    typedef cub::BlockReduce<double, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double localMax = 0.0;
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        localMax = max(localMax, input[idx]);
    }

    double maxVal = BlockReduce(temp_storage).Reduce(localMax, cub::Max());
    if (threadIdx.x == 0) *result = maxVal;
}

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1)*(y2 - y1)/(x2 - x1));
}

void initializeMatrix(std::vector<double>& M, int N) {
    M.assign(N * N, 0.0);
    M[0] = 10.0;
    M[N - 1] = 20.0;
    M[(N - 1) * N] = 20.0;
    M[(N - 1) * N + N - 1] = 30.0;
    for (int i = 1; i < N - 1; ++i) {
        M[i] = linearInterpolation(i, 0.0, M[0], N - 1, M[N - 1]);
        M[i * N] = linearInterpolation(i, 0.0, M[0], N - 1, M[(N - 1) * N]);
        M[i * N + N - 1] = linearInterpolation(i, 0.0, M[N - 1], N - 1, M[(N - 1) * N + N - 1]);
        M[(N - 1) * N + i] = linearInterpolation(i, 0.0, M[(N - 1) * N], N - 1, M[(N - 1) * N + N - 1]);
    }
}

int main(int argc, char const *argv[]) {
    opt::options_description desc("Опции");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Точность")
        ("size", opt::value<int>()->default_value(1024), "Размер матрицы")
        ("iterations", opt::value<int>()->default_value(1000000), "Количество итераций")
        ("help", "Помощь");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int N = vm["size"].as<int>();
    double tol = vm["accuracy"].as<double>();
    int maxIters = vm["iterations"].as<int>();
    const int errFreq = 10000;

    std::vector<double> h_prev, h_curr;
    initializeMatrix(h_prev, N);
    initializeMatrix(h_curr, N);

    double *d_prev, *d_curr;
    cudaMalloc(&d_prev, N * N * sizeof(double));
    cudaMalloc(&d_curr, N * N * sizeof(double));
    cudaMemcpy(d_prev, h_prev.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, h_curr.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(32, 16);
    dim3 grid((N - 2 + block.x - 1) / block.x,
              (N - 2 + block.y - 1) / block.y);
    int numBlocks = grid.x * grid.y;

    double *d_blockMax, *d_globalMax;
    cudaMalloc(&d_blockMax, numBlocks * sizeof(double));
    cudaMalloc(&d_globalMax, sizeof(double));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool graphCreated = false;

    double err = 1.0;
    int iter = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (iter < maxIters && err > tol) {
        if (!graphCreated) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            updateKernel<<<grid, block, 0, stream>>>(d_prev, d_curr, N);
            errorKernel<<<grid, block, 0, stream>>>(d_curr, d_prev, d_blockMax, N);
            finalMaxReduceKernel<<<1, 1024, 0, stream>>>(d_blockMax, d_globalMax, numBlocks);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        if (iter % errFreq == 0) {
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);

            cudaMemcpy(&err, d_globalMax, sizeof(double), cudaMemcpyDeviceToHost);
        } else {
            updateKernel<<<grid, block>>>(d_prev, d_curr, N);
        }

        std::swap(d_prev, d_curr);
        ++iter;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Time: " << ms << " ms, Iterations: " << iter << ", Final Error: " << err << std::endl;

    cudaMemcpy(h_curr.data(), d_curr, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_blockMax);
    cudaFree(d_globalMax);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);

    return 0;
}
