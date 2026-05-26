#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define NUM_RUNS 3

// Размеры матриц для тестирования
const std::vector<int> MATRIX_SIZES = { 100, 200, 400, 800, 1000, 1600, 2000 };

// Ядро CUDA: умножение матриц на GPU
template <typename T>
__global__ void matmul_gpu(T* A, T* B, T* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        T sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Умножение матриц на CPU
template <typename T>
void matmul_cpu(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Умножение матриц на GPU: копирование данных, запуск ядра, получение результата
template <typename T>
bool matmul_with_cuda(const std::vector<T>& h_A, const std::vector<T>& h_B, std::vector<T>& h_C, int N, float& kernel_time_ms) {
    T* d_A, * d_B, * d_C;
    size_t size = N * N * sizeof(T);

    // Выделение памяти на GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Настройка размеров блоков и сетки
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // События для точного измерения времени выполнения ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_gpu << <blocks, threads >> > (d_A, d_B, d_C, N);

    // Проверка ошибок после запуска ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Ожидание завершения всех нитей перед измерением времени
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    // Копирование результата обратно на хост
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Освобождение памяти на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return true;
}

// Проверка корректности результата
template <typename T>
bool check_result(const std::vector<T>& cpu_result, const std::vector<T>& gpu_result) {
    for (size_t i = 0; i < cpu_result.size(); i++) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-3) return false;
    }
    return true;
}

// Заполнение матрицы случайными значениями
template <typename T>
void fill_random(std::vector<T>& matrix) {
    std::mt19937 gen(42);  
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<int> dist(0, 10);
        for (auto& x : matrix) x = dist(gen);
    }
    else {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& x : matrix) x = static_cast<T>(dist(gen));
    }
}

int main() {
    // Открытие файла для записи результатов
    std::ofstream file("results.csv");
    file << "type,size,cpu_time_ms,gpu_time_ms,speedup\n";

    // Тестирование для типа int
    std::cout << "\n=== Testing int ===\n";
    for (int N : MATRIX_SIZES) {
        std::vector<int> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
        fill_random(A);
        fill_random(B);

        double cpu_total = 0, gpu_total = 0;
        bool kernel_ok = true;

        // Многократные прогоны для усреднения
        for (int run = 0; run < NUM_RUNS; run++) {
            // Измерение времени на CPU
            auto start = std::chrono::high_resolution_clock::now();
            matmul_cpu(A, B, C_cpu, N);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_total += std::chrono::duration<double, std::milli>(end - start).count();

            // Измерение времени на GPU
            float gpu_ms;
            bool ok = matmul_with_cuda(A, B, C_gpu, N, gpu_ms);
            if (!ok) kernel_ok = false;
            gpu_total += gpu_ms;
        }

        double cpu_avg = cpu_total / NUM_RUNS;
        double gpu_avg = gpu_total / NUM_RUNS;
        double speedup = cpu_avg / gpu_avg;
        bool correct = check_result(C_cpu, C_gpu);

        std::cout << N << "x" << N << ": CPU=" << cpu_avg << "ms GPU=" << gpu_avg << "ms Speedup=" << speedup << " Correct=" << (correct ? "YES" : "NO");
        if (!kernel_ok) std::cout << " KERNEL_ERROR";
        std::cout << "\n";

        file << "int," << N << "," << cpu_avg << "," << gpu_avg << "," << speedup << "\n";
    }

    // Тестирование для типа float
    std::cout << "\n=== Testing float ===\n";
    for (int N : MATRIX_SIZES) {
        std::vector<float> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
        fill_random(A);
        fill_random(B);

        double cpu_total = 0, gpu_total = 0;
        bool kernel_ok = true;

        for (int run = 0; run < NUM_RUNS; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            matmul_cpu(A, B, C_cpu, N);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_total += std::chrono::duration<double, std::milli>(end - start).count();

            float gpu_ms;
            bool ok = matmul_with_cuda(A, B, C_gpu, N, gpu_ms);
            if (!ok) kernel_ok = false;
            gpu_total += gpu_ms;
        }

        double cpu_avg = cpu_total / NUM_RUNS;
        double gpu_avg = gpu_total / NUM_RUNS;
        double speedup = cpu_avg / gpu_avg;
        bool correct = check_result(C_cpu, C_gpu);

        std::cout << N << "x" << N << ": CPU=" << cpu_avg << "ms GPU=" << gpu_avg << "ms Speedup=" << speedup << " Correct=" << (correct ? "YES" : "NO");
        if (!kernel_ok) std::cout << " KERNEL_ERROR";
        std::cout << "\n";

        file << "float," << N << "," << cpu_avg << "," << gpu_avg << "," << speedup << "\n";
    }

    // Тестирование для типа double
    std::cout << "\n=== Testing double ===\n";
    for (int N : MATRIX_SIZES) {
        std::vector<double> A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
        fill_random(A);
        fill_random(B);

        double cpu_total = 0, gpu_total = 0;
        bool kernel_ok = true;

        for (int run = 0; run < NUM_RUNS; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            matmul_cpu(A, B, C_cpu, N);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_total += std::chrono::duration<double, std::milli>(end - start).count();

            float gpu_ms;
            bool ok = matmul_with_cuda(A, B, C_gpu, N, gpu_ms);
            if (!ok) kernel_ok = false;
            gpu_total += gpu_ms;
        }

        double cpu_avg = cpu_total / NUM_RUNS;
        double gpu_avg = gpu_total / NUM_RUNS;
        double speedup = cpu_avg / gpu_avg;
        bool correct = check_result(C_cpu, C_gpu);

        std::cout << N << "x" << N << ": CPU=" << cpu_avg << "ms GPU=" << gpu_avg << "ms Speedup=" << speedup << " Correct=" << (correct ? "YES" : "NO");
        if (!kernel_ok) std::cout << " KERNEL_ERROR";
        std::cout << "\n";

        file << "double," << N << "," << cpu_avg << "," << gpu_avg << "," << speedup << "\n";
    }

    file.close();
    std::cout << "\nResults saved to results.csv\n";
    return 0;
}