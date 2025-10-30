#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>

#define N 2048
#define MASK_WIDTH 5
#define TILE_SIZE 256  // threads per block

// -------------------------
// GPU Kernel
// -------------------------
__global__ void Convolution1D_tiled(const float *input, const float *mask, float *output, int n) {
    __shared__ float sharedMem[TILE_SIZE + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int start = blockIdx.x * blockDim.x;
    int halo = MASK_WIDTH / 2;

    // Load shared memory with halos
    int inputIdx = start + tx - halo;
    if (inputIdx < 0)
        sharedMem[tx] = 0.0f;
    else if (inputIdx >= n)
        sharedMem[tx] = 0.0f;
    else
        sharedMem[tx] = input[inputIdx];

    // Load right halo elements
    if (tx < MASK_WIDTH - 1) {
        int rightIdx = start + blockDim.x + tx - halo;
        if (rightIdx < n)
            sharedMem[blockDim.x + tx] = input[rightIdx];
        else
            sharedMem[blockDim.x + tx] = 0.0f;
    }

    __syncthreads();

    // Compute convolution
    if (start + tx < n) {
        float sum = 0.0f;
        for (int j = 0; j < MASK_WIDTH; j++)
            sum += sharedMem[tx + j] * mask[j];
        output[start + tx] = sum;
    }
}

// -------------------------
// Host Function
// -------------------------
void BasicConvolution(float *input, float *mask, float *output, int n) {
    float *d_input, *d_mask, *d_output;
    size_t sizeInput = n * sizeof(float);
    size_t sizeMask = MASK_WIDTH * sizeof(float);
    float total_time = 0;
    struct timeval start, end;
    cudaMalloc(&d_input, sizeInput);
    cudaMalloc(&d_mask, sizeMask);
    cudaMalloc(&d_output, sizeInput);

    cudaMemcpy(d_input, input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeMask, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE);
    gettimeofday(&start, NULL);
    Convolution1D_tiled<<<gridSize, blockSize>>>(d_input, d_mask, d_output, n);
    gettimeofday(&end, NULL);
    cudaMemcpy(output, d_output, sizeInput, cudaMemcpyDeviceToHost);
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    printf("GPU latency: %.3f ms\n",  total_time );
}

// -------------------------
// CPU Reference
// -------------------------
void Convolution1D_CPU(float *input, float *mask, float *output, int n) {
    int halo = MASK_WIDTH / 2;
    float total_time = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < MASK_WIDTH; j++) {
            int idx = i + j - halo;
            if (idx >= 0 && idx < n)
                sum += input[idx] * mask[j];
        }
        output[i] = sum;
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    printf("CPU latency: %.3f ms\n",  total_time );
}

// -------------------------
// Main Evaluation
// -------------------------
int main() {
    float *input = (float *)malloc(N * sizeof(float));
    float *mask = (float *)malloc(MASK_WIDTH * sizeof(float));
    float *outputGPU = (float *)malloc(N * sizeof(float));
    float *outputCPU = (float *)malloc(N * sizeof(float));

    // Generate random input and mask
    for (int i = 0; i < N; i++)
        input[i] = (float)(rand() % 10);
    for (int i = 0; i < MASK_WIDTH; i++)
        mask[i] = (float)(rand() % 5);

    // CPU reference
    Convolution1D_CPU(input, mask, outputCPU, N);

    // GPU
    BasicConvolution(input, mask, outputGPU, N);

    // Compare results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(outputCPU[i] - outputGPU[i]) > 1e-3) {
            errors++;
        }
    }

    if (errors == 0)
        printf("✅ Convolution results match!\n");
    else
        printf("❌ %d mismatches found!\n", errors);

    free(input);
    free(mask);
    free(outputGPU);
    free(outputCPU);
    return 0;
}
