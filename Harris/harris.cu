#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <cmath>

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE 16

// Текстура для доступа к изображению
texture<unsigned char, 2, cudaReadModeElementType> texInput;

// Вычисление производных Ix и Iy
__global__ void computeDerivatives(float* Ix, float* Iy, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int x_left = (x > 0) ? x - 1 : x;
        int x_right = (x < width - 1) ? x + 1 : x;
        int y_top = (y > 0) ? y - 1 : y;
        int y_bot = (y < height - 1) ? y + 1 : y;

        unsigned char left = tex2D(texInput, x_left, y);
        unsigned char right = tex2D(texInput, x_right, y);
        unsigned char top = tex2D(texInput, x, y_top);
        unsigned char bottom = tex2D(texInput, x, y_bot);

        int idx = y * width + x;
        Ix[idx] = (right - left) * 0.5f;
        Iy[idx] = (bottom - top) * 0.5f;
    }
}

// Вычисление компонент матрицы Ixx, Ixy, Iyy
__global__ void computeMatrixComponents(float* Ix, float* Iy, float* Ixx, float* Ixy, float* Iyy, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        Ixx[idx] = Ix[idx] * Ix[idx];
        Ixy[idx] = Ix[idx] * Iy[idx];
        Iyy[idx] = Iy[idx] * Iy[idx];
    }
}

// Гауссово размытие
__global__ void gaussianBlur(float* input, float* output, int width, int height, float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        float wsum = 0.0f;
        int radius = (int)(3 * sigma);

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int nx = x + dx;
                int ny = y + dy;

                if (nx < 0) nx = 0;
                if (nx >= width) nx = width - 1;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;

                float w = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                wsum += w;
                sum += input[ny * width + nx] * w;
            }
        }

        output[y * width + x] = sum / wsum;
    }
}

// Вычисление Harris отклика R = det(A) - alpha * trace^2(A)
__global__ void computeHarrisResponse(float* A, float* B, float* C, float* R, int width, int height, float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        float det = A[idx] * C[idx] - B[idx] * B[idx];
        float trace = A[idx] + C[idx];
        R[idx] = det - alpha * trace * trace;
    }
}

// Пороговая обработка и подавление немаксимумов
__global__ void thresholdAndNonMaxSuppress(float* R, unsigned char* output, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        // По умолчанию не угол
        output[idx] = 0;

        if (R[idx] > threshold)
        {
            bool isMax = true;

            for (int dy = -1; dy <= 1 && isMax; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    // Пропустить текущий пиксель
                    if (dx == 0 && dy == 0) continue;

                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    {
                        if (R[ny * width + nx] >= R[idx])
                        {
                            isMax = false;
                            break;
                        }
                    }
                }
            }

            if (isMax)
            {
                output[idx] = 255;
            }
        }
    }
}

// Преобразование grayscale в RGB с выделением углов красным цветом
void convertToRGBWithCorners(unsigned char* grayscale, unsigned char* rgbOutput, unsigned char* corners, int width, int height)
{
    for (int i = 0; i < width * height; i++)
    {
        if (corners[i] == 255)
        {
            // Красный цвет для углов
            rgbOutput[i * 3 + 0] = 255;
            rgbOutput[i * 3 + 1] = 0;
            rgbOutput[i * 3 + 2] = 0;
        }
        else
        {
            // Исходное изображение (оттенки серого)
            unsigned char value = grayscale[i];
            rgbOutput[i * 3 + 0] = value;
            rgbOutput[i * 3 + 1] = value;
            rgbOutput[i * 3 + 2] = value;
        }
    }
}

// Вычисление RMSE между CPU и GPU результатами
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

// Вычисление количества совпадающих углов
int countMatchingCorners(unsigned char* cpuCorners, unsigned char* gpuCorners, int size)
{
    int matchCount = 0;
    int cpuCount = 0;
    int gpuCount = 0;

    for (int i = 0; i < size; i++)
    {
        if (cpuCorners[i] == 255) cpuCount++;
        if (gpuCorners[i] == 255) gpuCount++;
        if (cpuCorners[i] == 255 && gpuCorners[i] == 255) matchCount++;
    }

    std::cout << "CPU corners detected: " << cpuCount << std::endl;
    std::cout << "GPU corners detected: " << gpuCount << std::endl;

    return matchCount;
}

// GPU реализация детектора углов Харриса
void harrisGPU(unsigned char* h_input, unsigned char* h_output, int width, int height, float threshold, float sigma, float alpha)
{
    size_t imgSize = width * height * sizeof(unsigned char);
    size_t floatSize = width * height * sizeof(float);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_Ix = nullptr;
    float* d_Iy = nullptr;
    float* d_Ixx = nullptr;
    float* d_Ixy = nullptr;
    float* d_Iyy = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_R = nullptr;

    size_t pitch;

    // Выделение памяти
    cudaMallocPitch(&d_input, &pitch, width * sizeof(unsigned char), height);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_Ix, floatSize);
    cudaMalloc(&d_Iy, floatSize);
    cudaMalloc(&d_Ixx, floatSize);
    cudaMalloc(&d_Ixy, floatSize);
    cudaMalloc(&d_Iyy, floatSize);
    cudaMalloc(&d_A, floatSize);
    cudaMalloc(&d_B, floatSize);
    cudaMalloc(&d_C, floatSize);
    cudaMalloc(&d_R, floatSize);

    // Обнуление выходного массива
    cudaMemset(d_output, 0, imgSize);

    // Копирование данных
    cudaMemcpy2D(d_input, pitch, h_input, width * sizeof(unsigned char),
        width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // Настройка текстуры
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    texInput.addressMode[0] = cudaAddressModeClamp;
    texInput.addressMode[1] = cudaAddressModeClamp;
    texInput.filterMode = cudaFilterModePoint;
    texInput.normalized = false;

    // Привязка текстуры
    cudaBindTexture2D(NULL, texInput, d_input, channelDesc, width, height, pitch);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Вычисление производных
    computeDerivatives <<<gridSize, blockSize>>> (d_Ix, d_Iy, width, height);
    cudaDeviceSynchronize();

    // Вычисление компонент матрицы
    computeMatrixComponents <<<gridSize, blockSize>>> (d_Ix, d_Iy, d_Ixx, d_Ixy, d_Iyy, width, height);
    cudaDeviceSynchronize();

    // Гауссово размытие
    gaussianBlur <<<gridSize, blockSize>>> (d_Ixx, d_A, width, height, sigma);
    gaussianBlur <<<gridSize, blockSize>>> (d_Ixy, d_B, width, height, sigma);
    gaussianBlur <<<gridSize, blockSize>>> (d_Iyy, d_C, width, height, sigma);
    cudaDeviceSynchronize();

    // Вычисление Harris отклика
    computeHarrisResponse <<<gridSize, blockSize>>> (d_A, d_B, d_C, d_R, width, height, alpha);
    cudaDeviceSynchronize();

    // Пороговая обработка и подавление немаксимумов
    thresholdAndNonMaxSuppress <<<gridSize, blockSize>>> (d_R, d_output, width, height, threshold);
    cudaDeviceSynchronize();

    // Копирование результата обратно
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Отвязывание текстуры
    cudaUnbindTexture(texInput);

    // Очистка памяти
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_Ixx);
    cudaFree(d_Ixy);
    cudaFree(d_Iyy);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_R);
}

// CPU реализация детектора углов Харриса
void harrisCPU(unsigned char* input, unsigned char* output, int width, int height, float threshold, float sigma, float alpha)
{
    int size = width * height;

    float* Ix = new float[size];
    float* Iy = new float[size];
    float* Ixx = new float[size];
    float* Ixy = new float[size];
    float* Iyy = new float[size];
    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];
    float* R = new float[size];

    // Инициализация выходного массива
    memset(output, 0, size);

    // Вычисление производных
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            unsigned char c = input[idx];
            unsigned char l = (x > 0) ? input[idx - 1] : c;
            unsigned char r = (x < width - 1) ? input[idx + 1] : c;
            unsigned char t = (y > 0) ? input[(y - 1) * width + x] : c;
            unsigned char b = (y < height - 1) ? input[(y + 1) * width + x] : c;

            Ix[idx] = (r - l) * 0.5f;
            Iy[idx] = (b - t) * 0.5f;
        }
    }

    // Вычисление компонент матрицы
    for (int i = 0; i < size; i++)
    {
        Ixx[i] = Ix[i] * Ix[i];
        Ixy[i] = Ix[i] * Iy[i];
        Iyy[i] = Iy[i] * Iy[i];
    }

    // Гауссово размытие
    int radius = (int)(3 * sigma);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sumA = 0, sumB = 0, sumC = 0, wsum = 0;
            int idx = y * width + x;

            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;

                    float w = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                    wsum += w;
                    sumA += Ixx[ny * width + nx] * w;
                    sumB += Ixy[ny * width + nx] * w;
                    sumC += Iyy[ny * width + nx] * w;
                }
            }

            A[idx] = sumA / wsum;
            B[idx] = sumB / wsum;
            C[idx] = sumC / wsum;
        }
    }

    // Вычисление Harris отклика
    for (int i = 0; i < size; i++)
    {
        float det = A[i] * C[i] - B[i] * B[i];
        float trace = A[i] + C[i];
        R[i] = det - alpha * trace * trace;
    }

    // Пороговая обработка и подавление немаксимумов
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            if (R[idx] > threshold)
            {
                bool isMax = true;

                for (int dy = -1; dy <= 1 && isMax; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;

                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            if (R[ny * width + nx] >= R[idx])
                            {
                                isMax = false;
                                break;
                            }
                        }
                    }
                }

                if (isMax)
                {
                    output[idx] = 255;
                }
            }
        }
    }

    delete[] Ix;
    delete[] Iy;
    delete[] Ixx;
    delete[] Ixy;
    delete[] Iyy;
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] R;
}

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " <input_image> <threshold> <output_cpu.jpg> <output_gpu.jpg>" << std::endl;
        std::cout << "Recommended threshold: 5000000" << std::endl;
        return -1;
    }

    int w, h, ch;
    unsigned char* img = stbi_load(argv[1], &w, &h, &ch, 1);

    if (!img)
    {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }

    float threshold = atof(argv[2]);
    const char* outCpu = argv[3];
    const char* outGpu = argv[4];

    std::cout << "Image: " << w << "x" << h << " Threshold: " << threshold << std::endl;

    unsigned char* cpuCorners = new unsigned char[w * h];
    unsigned char* gpuCorners = new unsigned char[w * h];

    // Массивы для RGB изображений с красными углами
    unsigned char* cpuRgb = new unsigned char[w * h * 3];
    unsigned char* gpuRgb = new unsigned char[w * h * 3];

    // Выполнение CPU версии
    std::cout << "\n--- CPU Processing ---" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    harrisCPU(img, cpuCorners, w, h, threshold, 1.0f, 0.04f);
    auto t2 = std::chrono::high_resolution_clock::now();
    long long cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

    // Выполнение GPU версии
    std::cout << "\n--- GPU Processing ---" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    harrisGPU(img, gpuCorners, w, h, threshold, 1.0f, 0.04f);
    t2 = std::chrono::high_resolution_clock::now();
    long long gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;

    // Сравнение точности
    std::cout << "\n--- Accuracy Comparison ---" << std::endl;

    double rmse = calculateRMSE(cpuCorners, gpuCorners, w * h);
    std::cout << "RMSE (CPU vs GPU): " << rmse << std::endl;

    int matchingCorners = countMatchingCorners(cpuCorners, gpuCorners, w * h);
    double matchRate = (double)matchingCorners / std::max(
        std::count(cpuCorners, cpuCorners + w * h, 255),
        std::count(gpuCorners, gpuCorners + w * h, 255)) * 100.0;
    std::cout << "Matching corners: " << matchingCorners << std::endl;
    std::cout << "Corner match rate: " << matchRate << "%" << std::endl;

    // Ускорение
    std::cout << "\n--- Speedup ---" << std::endl;
    std::cout << "Speedup (CPU/GPU): " << (float)cpuTime / (float)gpuTime << "x" << std::endl;

    // Преобразование в RGB с выделением углов красным цветом
    convertToRGBWithCorners(img, cpuRgb, cpuCorners, w, h);
    convertToRGBWithCorners(img, gpuRgb, gpuCorners, w, h);

    // Сохранение результатов
    stbi_write_jpg(outCpu, w, h, 3, cpuRgb, 100);
    stbi_write_jpg(outGpu, w, h, 3, gpuRgb, 100);

    std::cout << "\nSaved: " << outCpu << " and " << outGpu << std::endl;
    std::cout << "Corners are highlighted in RED" << std::endl;

    // Очистка памяти
    stbi_image_free(img);
    delete[] cpuCorners;
    delete[] gpuCorners;
    delete[] cpuRgb;
    delete[] gpuRgb;

    return 0;
}