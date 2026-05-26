#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>

const size_t THREADS_PER_BLOCK = 256;
const size_t REPEAT_COUNT = 10;
const std::vector<size_t> VECTOR_SIZES = { 1000, 5000, 10000, 50000, 100000, 500000, 1000000 };

struct ExperimentResult {
    size_t size;
    std::string dataType;
    double cpuTimeMs;
    double gpuTimeMs;
    double gpuKernelMs;
    double speedup;
    double cpuSum;
    double gpuSum;
};

// Сложение вектора на CPU
template<typename T>
T vector_sum_cpu(const T* vec, size_t length) {
    T sum = 0;
    for (size_t i = 0; i < length; ++i)
        sum += vec[i];
    return sum;
}

// Ядро CUDA для редукции
template<typename T>
__global__ void vector_sum_kernel(const T* input, T* partial_sums, size_t length) {
    extern __shared__ T shared_data[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[threadIdx.x] = (idx < length) ? input[idx] : T(0);
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        partial_sums[blockIdx.x] = shared_data[0];
}

// Сложение вектора на GPU
template<typename T>
T vector_sum_gpu(const std::vector<T>& h_input, double* kernel_time = nullptr) {
    size_t length = h_input.size();
    size_t grid_size = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    T* d_input, * d_partial_sums;
    cudaMalloc(&d_input, length * sizeof(T));
    cudaMalloc(&d_partial_sums, grid_size * sizeof(T));

    cudaMemcpy(d_input, h_input.data(), length * sizeof(T), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    vector_sum_kernel<T> << <grid_size, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(T) >> > (d_input, d_partial_sums, length);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (kernel_time)
        *kernel_time += std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<T> h_partial_sums(grid_size);
    cudaMemcpy(h_partial_sums.data(), d_partial_sums, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

    T result = 0;
    for (size_t i = 0; i < grid_size; ++i)
        result += h_partial_sums[i];

    cudaFree(d_input);
    cudaFree(d_partial_sums);

    return result;
}

// Заполнение вектора случайными числами
template<typename T>
void fill_vector_random(std::vector<T>& vec) {
    std::mt19937 gen(42);
    if constexpr (std::is_same<T, int>::value) {
        std::uniform_int_distribution<int> dist(1, 100);
        for (auto& x : vec)
            x = dist(gen);
    }
    else {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : vec)
            x = static_cast<T>(dist(gen));
    }
}

// Проверка корректности результатов
template<typename T>
bool verify_results(const std::vector<T>& cpu_result, const std::vector<T>& gpu_result) {
    if (cpu_result.size() != gpu_result.size())
        return false;

    for (size_t i = 0; i < cpu_result.size(); ++i) {
        T diff = cpu_result[i] - gpu_result[i];
        if (diff < 0) diff = -diff;
        if (diff > T(1e-3))
            return false;
    }
    return true;
}

// Запуск экспериментов для одного типа данных
template<typename T>
void run_experiments(const std::string& type_name, std::vector<ExperimentResult>& results) {
    std::cout << "\n========== " << type_name << " ==========" << std::endl;
    std::cout << std::left << std::setw(12) << "Size"
        << std::setw(12) << "CPU(ms)"
        << std::setw(12) << "GPU(ms)"
        << std::setw(15) << "GPUker(ms)"
        << std::setw(10) << "Speedup"
        << std::setw(15) << "CPU_sum"
        << std::setw(15) << "GPU_sum"
        << std::setw(10) << "Correct" << std::endl;

    for (size_t size : VECTOR_SIZES) {
        std::vector<T> h_vector(size);
        fill_vector_random<T>(h_vector);

        double cpu_total = 0.0, gpu_total = 0.0, gpu_kernel_total = 0.0;
        T cpu_sum = 0, gpu_sum = 0;

        for (size_t rep = 0; rep < REPEAT_COUNT; ++rep) {
            auto start = std::chrono::high_resolution_clock::now();
            cpu_sum = vector_sum_cpu<T>(h_vector.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_total += std::chrono::duration<double, std::milli>(end - start).count();

            gpu_sum = vector_sum_gpu<T>(h_vector, &gpu_kernel_total);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            gpu_total += std::chrono::duration<double, std::milli>(gpu_end - end).count();
        }

        double cpu_avg = cpu_total / REPEAT_COUNT;
        double gpu_avg = gpu_total / REPEAT_COUNT;
        double gpu_kernel_avg = gpu_kernel_total / REPEAT_COUNT;
        double speedup = cpu_avg / gpu_avg;

        ExperimentResult res;
        res.size = size;
        res.dataType = type_name;
        res.cpuTimeMs = cpu_avg;
        res.gpuTimeMs = gpu_avg;
        res.gpuKernelMs = gpu_kernel_avg;
        res.speedup = speedup;
        res.cpuSum = static_cast<double>(cpu_sum);
        res.gpuSum = static_cast<double>(gpu_sum);
        results.push_back(res);

        bool correct = (cpu_sum == gpu_sum);

        std::cout << std::fixed << std::setprecision(2)
            << std::setw(12) << size
            << std::setw(12) << cpu_avg
            << std::setw(12) << gpu_avg
            << std::setw(15) << gpu_kernel_avg
            << std::setw(10) << speedup
            << std::setw(15) << static_cast<double>(cpu_sum)
            << std::setw(15) << static_cast<double>(gpu_sum)
            << std::setw(10) << (correct ? "YES" : "NO") << std::endl;
    }
}

void save_results_to_csv(const std::vector<ExperimentResult>& results) {
    std::ofstream file("vector_sum_results_double.csv");
    file << "Size,DataType,CPU_Time_ms,GPU_Time_ms,GPU_Kernel_ms,Speedup,CPU_Sum,GPU_Sum\n";

    for (const auto& res : results) {
        file << res.size << ","
            << res.dataType << ","
            << res.cpuTimeMs << ","
            << res.gpuTimeMs << ","
            << res.gpuKernelMs << ","
            << res.speedup << ","
            << res.cpuSum << ","
            << res.gpuSum << "\n";
    }
    file.close();
    std::cout << "\nResults saved to vector_sum_results_double.csv" << std::endl;
}

int main() {
    std::vector<ExperimentResult> all_results;

    // run_experiments<int>("int", all_results);
    //run_experiments<float>("float", all_results);
     run_experiments<double>("double", all_results);

    save_results_to_csv(all_results);

    return 0;
}