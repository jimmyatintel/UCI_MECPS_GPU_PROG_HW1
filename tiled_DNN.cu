#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>

#define BATCH 1
#define C_IN 4
#define D_H 256
#define D_W 256
#define M_OUT 16
#define D_K 3
#define TILE 16

const int H_out = D_H - D_K + 1;
const int W_out = D_W - D_K + 1;

// ----------------- GPU Kernel (tiled, shared memory) -----------------
__global__ void conv2d_tiled_valid(
    const float *  input,
    const float *  weight,
    const float *  bias,
    float *output,
    int C, int H, int W, int K, int M)
{
    int m = blockIdx.z;                        // output feature map
    int tile_origin_x = blockIdx.x * TILE;
    int tile_origin_y = blockIdx.y * TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_x = tile_origin_x + tx;
    int out_y = tile_origin_y + ty;

    int tile_size = TILE + K - 1;
    extern __shared__ float s_tile[];

    float acc = 0.0f;

    for (int c = 0; c < C; ++c) {
        // Load tile + halo into shared memory
        for (int y = ty; y < tile_size; y += blockDim.y) {
            for (int x = tx; x < tile_size; x += blockDim.x) {
                int in_y = tile_origin_y + y;
                int in_x = tile_origin_x + x;
                s_tile[y * tile_size + x] = input[c * (H * W) + in_y * W + in_x];
            }
        }
        __syncthreads();

        // Compute convolution for this thread's output
        if (out_y < H_out && out_x < W_out) {
            const float *wptr = weight + (m * C + c) * (K * K);
            for (int p = 0; p < K; ++p) {
                int srow = ty + p;
                int base = srow * tile_size + tx;
                for (int q = 0; q < K; ++q) {
                    acc += s_tile[base + q] * wptr[p*K + q];
                }
            }
        }
        __syncthreads();
    }

    if (out_y < H_out && out_x < W_out) {
        output[m * (H_out * W_out) + out_y * W_out + out_x] = acc + bias[m];
    }
}

// ----------------- Host function -----------------
void ConvolutionLayer(const float *h_input, const float *h_weight, const float *h_bias, float *h_output, int C, int H, int W, int K, int M)
{
    size_t szInput  = (size_t)C * H * W * sizeof(float);
    size_t szWeight = (size_t)M * C * K * K * sizeof(float);
    size_t szBias   = (size_t)M * sizeof(float);
    size_t szOutput = (size_t)M * H_out * W_out * sizeof(float);
    struct timeval start, end;
    float total_time = 0;
    float *d_input = NULL, *d_weight = NULL, *d_bias = NULL, *d_output = NULL;
    cudaMalloc((void**)&d_input, szInput);
    cudaMalloc((void**)&d_weight, szWeight);
    cudaMalloc((void**)&d_bias, szBias);
    cudaMalloc((void**)&d_output, szOutput);

    cudaMemcpy(d_input, h_input, szInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, szWeight, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, szBias, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE, 1);
    dim3 gridDim( (W_out + TILE - 1) / TILE, (H_out + TILE - 1) / TILE, M );

    int shared_bytes = (TILE + K - 1) * (TILE + K - 1) * sizeof(float);
    gettimeofday(&start, NULL);
    conv2d_tiled_valid<<<gridDim, blockDim, shared_bytes>>>(d_input, d_weight, d_bias, d_output, C, H, W, K, M);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaMemcpy(h_output, d_output, szOutput, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    printf("GPU latency: %.3f ms\n",  total_time );
}

// ----------------- CPU reference -----------------
void ConvolutionLayer_CPU_Valid(const float *input, const float *weight, const float *bias, float *output,
                                int C, int H, int W, int K, int M)
{
    struct timeval start, end;
    float total_time = 0;
    gettimeofday(&start, NULL);
    for (int m = 0; m < M; ++m) {
        for (int y = 0; y < H_out; ++y) {
            for (int x = 0; x < W_out; ++x) {
                float sum = 0.0f;
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < K; ++p) {
                        for (int q = 0; q < K; ++q) {
                            float in_val = input[c * (H*W) + (y+p)*W + (x+q)];
                            float w = weight[(m*C + c) * (K*K) + p*K + q];
                            sum += in_val * w;
                        }
                    }
                }
                output[m * (H_out * W_out) + y * W_out + x] = sum + bias[m];
            }
        }
    }
    gettimeofday(&end, NULL);
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    printf("CPU latency: %.3f ms\n",  total_time );
}

// ----------------- Main: Test -----------------
int main() {
    srand(1234);

    size_t szInput  = (size_t)C_IN * D_H * D_W;
    size_t szWeight = (size_t)M_OUT * C_IN * D_K * D_K;
    size_t szBias   = (size_t)M_OUT;
    size_t szOutput = (size_t)M_OUT * H_out * W_out;

    float *h_input  = (float*)malloc(szInput * sizeof(float));
    float *h_weight = (float*)malloc(szWeight * sizeof(float));
    float *h_bias   = (float*)malloc(szBias   * sizeof(float));
    float *h_output_gpu = (float*)malloc(szOutput * sizeof(float));
    float *h_output_cpu = (float*)malloc(szOutput * sizeof(float));

    // random input / weight / bias
    for (size_t i = 0; i < szInput; ++i) h_input[i] = ((float)(rand()%10)-5)/8.0f;
    for (size_t i = 0; i < szWeight; ++i) h_weight[i] = ((float)(rand()%10)-5)/8.0f;
    for (size_t i = 0; i < szBias; ++i)   h_bias[i] = 0.0f;

    // CPU reference
    ConvolutionLayer_CPU_Valid(h_input, h_weight, h_bias, h_output_cpu, C_IN, D_H, D_W, D_K, M_OUT);

    // GPU
    ConvolutionLayer(h_input, h_weight, h_bias, h_output_gpu, C_IN, D_H, D_W, D_K, M_OUT);

    // compare
    int errors = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < szOutput; ++i) {
        double a = (double)h_output_cpu[i];
        double b = (double)h_output_gpu[i];
        double err = fabs(a-b);
        if (err > 1e-3) {
            errors++;
            if (errors < 20) {
                size_t m = i / (H_out*W_out);
                size_t rem = i % (H_out*W_out);
                size_t y = rem / W_out;
                size_t x = rem % W_out;
                printf("Mismatch m=%zu y=%zu x=%zu CPU=%.6f GPU=%.6f\n", m, y, x, a, b);
            }
        }
        if (err > max_err) max_err = err;
    }

    if (errors==0) printf("✅ PASS: GPU matches CPU (max abs err %.6f)\n", max_err);
    else printf("❌ FAIL: %d mismatches (max abs err %.6f)\n", errors, max_err);

    free(h_input); free(h_weight); free(h_bias); free(h_output_gpu); free(h_output_cpu);
    return 0;
}
