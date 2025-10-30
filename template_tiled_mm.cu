// Matrix Multiplication C = A * B
// Important: The 2-D Matrix is stored in 1-D array in row-order (Matrix[i][j] --> Array[i * Width + j])

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define DimMIN 100
#define DimMAX 100
#define TestSamples 1    //setup numbers of test samples
#define nIter 1 // Setup num of interations

__global__ void MatrixMulCUDA(
    const float *A, 
    const float *B, 
    float *C,
    int wA,
    int wB) 
    {
        // implement your CUDA kernel here
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int col = blockIdx.x * blockDim.x + threadIdx.x;

      // Each thread computes one element of C
      float value = 0.0f;
      if (row < wA && col < wB) {
        for (int k = 0; k < wA; ++k) {
            value += A[row * wA + k] * B[k * wB + col];
        }
        C[row * wB + col] = value;
      }
    }


void MatrixMultiplication(float *h_A, float *h_B, float *h_C, int wA, size_t mem_size_A, int wB, size_t mem_size_B, size_t mem_size_C) {
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, mem_size_A);
    cudaMalloc((void**)&d_B, mem_size_B);
    cudaMalloc((void**)&d_C, mem_size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // Set grid and block dimensions here
    int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((wB + BLOCK_SIZE - 1) / BLOCK_SIZE,(wA + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //create timer
    float total_time = 0;
    struct timeval start, end;
    for (int j = 0; j < nIter; j++) 
{
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    // Call the kernel
    MatrixMulCUDA<<<grid, threads>>>(d_A, d_B, d_C, wA, wB);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;

}
    //print kernel latency
    printf("GPU latency: %.3f ms\n",  total_time/nIter );

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(void) {
    //set random seed
    float total_time = 0;
    srand(time(0));
    for(int testiter = 0; testiter < TestSamples; testiter++)
{
//Print the No. of Test Sample
    printf(">>>>>>>>>>>>>>>>> test No. %d >>>>>>>>>>>>>>>>>\n",testiter+1);
  int minNum = DimMIN;
  int maxNum = DimMAX;
/*
dimsA.x is WidthA 
dimsA.y is HeightA    (Width x Height of Matrix A)
dimsA.x is WidthB 
dimsA.y is HeightB    (Width x Height of Matrix B)

*/
  dim3 dimsA(100, 100, 1);
  dim3 dimsB(100, 100, 1);
  // Set random input matrix size 
  dimsA.x = rand() % (maxNum - minNum + 1) + minNum;
  dimsA.y = rand() % (maxNum - minNum + 1) + minNum;
  dimsB.x = rand() % (maxNum - minNum + 1) + minNum;
  dimsB.y = dimsA.x;  // WidthA should be equal to HeightB
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  printf("[Matrix Multiplication of (%d,%d) x (%d,%d) ]\n", dimsA.y,dimsA.x,dimsB.y, dimsB.x);

  // Allocate the host input Matrix A
  unsigned int size_A = dimsA.x * dimsA.y;
  size_t mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  
  // Allocate the host input Matrix B
  unsigned int size_B = dimsB.x * dimsB.y;
  size_t mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // Allocate the host output Matrix C
  unsigned int size_C = dimsC.y * dimsC.x;
  size_t mem_size_C = sizeof(float) * size_C;
  float *h_C = (float *)malloc(mem_size_C);
  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors with random data
  for (int i = 0; i < size_A; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < size_B; ++i) {
    h_B[i] = rand() / (float)RAND_MAX;
  }

  //call CUDA top function here
  MatrixMultiplication(h_A, h_B, h_C, dimsA.x, mem_size_A, dimsB.x, mem_size_B, mem_size_C);

  //check results here
struct timeval start, end;
for(int i = 0; i<dimsA.y ;i++)
{
    for(int j = 0; j<dimsB.x ;j++)
    {
        gettimeofday(&start, NULL);
        float buffer = 0.;
        for(int k = 0; k<dimsA.x ;k++)
        {
            buffer += h_A[i*dimsA.x + k] * h_B[k*dimsB.x + j];
            // printf("A B: %.3f,  %.3f \n",  h_A[i*dimsA.x + k], h_B[k*dimsB.x + j] );
        }
        gettimeofday(&end, NULL);
        total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
        if (fabs(h_C[i*dimsB.x + j] - buffer) > 1e-4) {
            printf("CPU result: %.3f \n",  buffer );
            printf("h_C: %.3f \n",  h_C[i*dimsB.x + j] );
      fprintf(stderr, "Result verification failed at element (%d,%d)!\n", i,j);
      // count++;
      //exit(EXIT_FAILURE);
    }

    }
}
printf("CPU latency: %.3f ms\n",  total_time/nIter );
  free(h_A);
  free(h_B);
  free(h_C);
}
return 0;
}