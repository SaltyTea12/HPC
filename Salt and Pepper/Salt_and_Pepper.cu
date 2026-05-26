// Подключение библиотек
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <algorithm>

#define EASYBMP_IMPLEMENTATION
#include "EasyBMP/EasyBMP.h"

// Текстура для доступа к изображению
texture<unsigned char, 2, cudaReadModeElementType> texInput;

// Функция добавления шума salt-and-pepper
void addSaltPepperNoise(unsigned char* image, int width, int height, float saltProb, float pepperProb)
{
    for (int i = 0; i < width * height; i++)
    {
        float r = (float)rand() / RAND_MAX;
        if (r < saltProb)
            image[i] = 255;
        else if (r < saltProb + pepperProb)
            image[i] = 0;
    }
}

// Медианный фильтр на CPU
void medianFilterCPU(unsigned char* input, unsigned char* output, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char window[9];
            int idx = 0;

            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;

                    window[idx++] = input[ny * width + nx];
                }
            }

            std::sort(window, window + 9);
            output[y * width + x] = window[4];
        }
    }
}

// Ядро медианного фильтра на GPU с использованием texture memory
__global__ void medianFilterGPU(unsigned char* d_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        unsigned char window[9];
        int idx = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx < 0) nx = 0;
                if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;

                window[idx++] = tex2D(texInput, nx, ny);
            }
        }

        // Сортировка пузырьком
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8 - i; j++)
            {
                if (window[j] > window[j + 1])
                {
                    unsigned char temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        d_output[y * width + x] = window[4];
    }
}

// Вычисление RMSE
double calculateRMSE(unsigned char* img1, unsigned char* img2, int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        double diff = (double)img1[i] - (double)img2[i];
        sum += diff * diff;
    }
    return sqrt(sum / size);
}

// Конвертация BMP в grayscale
void bmpToGrayscale(BMP& bmp, unsigned char* array, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            RGBApixel pixel = bmp.GetPixel(x, y);
            array[y * width + x] = (unsigned char)(0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue);
        }
    }
}

// Конвертация grayscale в BMP
void grayscaleToBMP(unsigned char* array, BMP& bmp, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            RGBApixel pixel;
            unsigned char value = array[y * width + x];
            pixel.Red = value;
            pixel.Green = value;
            pixel.Blue = value;
            pixel.Alpha = 0;
            bmp.SetPixel(x, y, pixel);
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <input.bmp>\n", argv[0]);
        return 1;
    }

    // Загрузка изображения
    BMP originalImage;
    if (!originalImage.ReadFromFile(argv[1]))
    {
        printf("Error: Failed to read file\n");
        return 1;
    }

    int width = originalImage.TellWidth();
    int height = originalImage.TellHeight();
    int size = width * height;

    printf("Image size: %dx%d\n", width, height);

    // Выделение памяти
    unsigned char* h_original = (unsigned char*)malloc(size);
    unsigned char* h_noisy = (unsigned char*)malloc(size);
    unsigned char* h_cpuResult = (unsigned char*)malloc(size);
    unsigned char* h_gpuResult = (unsigned char*)malloc(size);

    // Конвертация в grayscale
    bmpToGrayscale(originalImage, h_original, width, height);

    // Добавление шума
    memcpy(h_noisy, h_original, size);
    addSaltPepperNoise(h_noisy, width, height, 0.05f, 0.05f);

    // Сохранение зашумленного изображения
    BMP noisyImage;
    noisyImage.SetSize(width, height);
    noisyImage.SetBitDepth(24);
    grayscaleToBMP(h_noisy, noisyImage, width, height);
    noisyImage.WriteToFile("noisy.bmp");

    // CPU Filter
    printf("\n--- CPU Processing ---\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    medianFilterCPU(h_noisy, h_cpuResult, width, height);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart);
    printf("CPU time: %lld ms\n", cpuTime.count());

    // Сохранение CPU результата
    BMP cpuImage;
    cpuImage.SetSize(width, height);
    cpuImage.SetBitDepth(24);
    grayscaleToBMP(h_cpuResult, cpuImage, width, height);
    cpuImage.WriteToFile("output_cpu.bmp");

    // GPU Filter
    printf("\n--- GPU Processing ---\n");

    // Выделение памяти на GPU с pitch для texture
    unsigned char* d_input = NULL;
    unsigned char* d_output = NULL;
    size_t pitch;
    cudaError_t cudaStatus;

    cudaStatus = cudaMallocPitch((void**)&d_input, &pitch, width * sizeof(unsigned char), height);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMallocPitch failed: %s\n", cudaGetErrorString(cudaStatus));
        free(h_original); free(h_noisy); free(h_cpuResult); free(h_gpuResult);
        return 1;
    }

    cudaStatus = cudaMalloc(&d_output, size);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        free(h_original); free(h_noisy); free(h_cpuResult); free(h_gpuResult);
        return 1;
    }

    // Копирование данных на GPU с pitch
    cudaStatus = cudaMemcpy2D(d_input, pitch, h_noisy, width * sizeof(unsigned char),
        width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMemcpy2D failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input); cudaFree(d_output);
        free(h_original); free(h_noisy); free(h_cpuResult); free(h_gpuResult);
        return 1;
    }

    // Настройка текстуры
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    texInput.addressMode[0] = cudaAddressModeClamp;
    texInput.addressMode[1] = cudaAddressModeClamp;
    texInput.filterMode = cudaFilterModePoint;
    texInput.normalized = false;

    // Привязка текстуры
    cudaStatus = cudaBindTexture2D(NULL, texInput, d_input, channelDesc, width, height, pitch);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaBindTexture2D failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input); cudaFree(d_output);
        free(h_original); free(h_noisy); free(h_cpuResult); free(h_gpuResult);
        return 1;
    }

    // Запуск ядра
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    auto gpuStart = std::chrono::high_resolution_clock::now();
    medianFilterGPU << <gridSize, blockSize >> > (d_output, width, height);
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    auto gpuEnd = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEnd - gpuStart);
    printf("GPU time: %lld ms\n", gpuTime.count());

    // Копирование результата обратно
    cudaStatus = cudaMemcpy(h_gpuResult, d_output, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaMemcpy back failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Отвязывание текстуры
    cudaUnbindTexture(texInput);

    // Сохранение GPU результата
    BMP gpuImage;
    gpuImage.SetSize(width, height);
    gpuImage.SetBitDepth(24);
    grayscaleToBMP(h_gpuResult, gpuImage, width, height);
    gpuImage.WriteToFile("output_gpu.bmp");

    // Выисление метрик
    printf("\n--- Quality Metrics ---\n");

    double rmseNoisy = calculateRMSE(h_original, h_noisy, size);
    printf("RMSE (Original vs Noisy): %.4f\n", rmseNoisy);

    double rmseCPU = calculateRMSE(h_original, h_cpuResult, size);
    printf("RMSE (Original vs CPU): %.4f\n", rmseCPU);

    double rmseGPU = calculateRMSE(h_original, h_gpuResult, size);
    printf("RMSE (Original vs GPU): %.4f\n", rmseGPU);

    double rmseCPUvsGPU = calculateRMSE(h_cpuResult, h_gpuResult, size);
    printf("RMSE (CPU vs GPU): %.4f\n", rmseCPUvsGPU);

    // Ускорение
    if (gpuTime.count() > 0)
    {
        double speedup = (double)cpuTime.count() / (double)gpuTime.count();
        printf("\n--- Speedup ---\n");
        printf("CPU time: %lld ms\n", cpuTime.count());
        printf("GPU time: %lld ms\n", gpuTime.count());
        printf("Speedup (CPU/GPU): %.2fx\n", speedup);
    }

    // Очистка памяти
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_original);
    free(h_noisy);
    free(h_cpuResult);
    free(h_gpuResult);

    printf("\nProcessing completed!\n");
    printf("Output files:\n");
    printf("  - noisy.bmp\n");
    printf("  - output_cpu.bmp\n");
    printf("  - output_gpu.bmp\n");

    return 0;
}