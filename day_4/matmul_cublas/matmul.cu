#include <cstdio>
#include <cublas_v2.h>

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

#define CHECK_CUBLAS(call)  \
  do {  \
    cublasStatus_t status_ = call;  \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__, cublasGetStatusName(status_), cublasGetStatusString(status_));  \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;
static cublasHandle_t handle;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // A_gpu = A^T (K X M)
  CHECK_CUBLAS(cublasSetMatrix(K, M, sizeof(float), _A, K, A_gpu, K));
  // B_gpu = B^T (N X K)
  CHECK_CUBLAS(cublasSetMatrix(N, K, sizeof(float), _B, N, B_gpu, N));
  // C_gpu = C^T = B^T * A^T (N X M)
  const float alpha = 1.0f, beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_gpu, N, A_gpu, K, &beta, C_gpu, N));
  // C = C^T^T = C_gpu^T (M X N)
  CHECK_CUBLAS(cublasGetMatrix(N, M, sizeof(float), C_gpu, N, _C, N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc((void **) &A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc((void **) &B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc((void **) &C_gpu, sizeof(float) * M * N));
  CHECK_CUBLAS(cublasCreate(&handle));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUBLAS(cublasDestroy(handle));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
