#include <cstdio>

#include "matmul.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Define kernel function
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_idx >= N || y_idx >= M) return;
  C[y_idx * M + x_idx] = 0.0f;
  float a0, a1, a2, a3, a4, a5, a6, a7;
  float b0, b1, b2, b3, b4, b5, b6, b7;
  int k;

  for (k = 0; k < (K/8) * 8; k+=8) {
    a0 = A[y_idx * K + k];
    a1 = A[y_idx * K + k + 1];
    a2 = A[y_idx * K + k + 2];
    a3 = A[y_idx * K + k + 3];
    a4 = A[y_idx * K + k + 4];
    a5 = A[y_idx * K + k + 5];
    a6 = A[y_idx * K + k + 6];
    a7 = A[y_idx * K + k + 7];

    b0 = B[k * N + x_idx];
    b1 = B[(k + 1) * N + x_idx];
    b2 = B[(k + 2) * N + x_idx];
    b3 = B[(k + 3) * N + x_idx];
    b4 = B[(k + 4) * N + x_idx];
    b5 = B[(k + 5) * N + x_idx];
    b6 = B[(k + 6) * N + x_idx];
    b7 = B[(k + 7) * N + x_idx];

    C[y_idx * M + x_idx] += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7;
  }
  for (; k < K; k++) {
    C[y_idx * M + x_idx] += A[y_idx * K + k] * B[k * N + x_idx];
  }
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // // Remove this line after you complete the matmul on GPU
  // naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(float) * M * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
