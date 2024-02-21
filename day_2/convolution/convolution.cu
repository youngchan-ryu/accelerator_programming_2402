#include <cstdio>

#include "convolution.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void naive_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = _I, *F = _F, *O = _O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }
}

static float *I_gpu, *F_gpu, *O_gpu;

// __global__ void convolution_kernel_1(float *I, float *F, float *O, int N, int C, int H, int W, int K, int R, int S, int ON, int OC, int OH, int OW, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
//   int on = blockIdx.x * blockDim.x + threadIdx.x;
//   int oc = blockIdx.y * blockDim.y + threadIdx.y;
//   if (on >= ON || oc >= OC) return;
//   for (int oh = 0; oh < OH; ++oh) {
//     for (int ow = 0; ow < OW; ++ow) {
//       float sum = 0;
//       for (int c = 0; c < C; ++c) {
//         for (int r = 0; r < R; ++r) {
//           for (int s = 0; s < S; ++s) {
//             const int n = on;
//             const int h = oh * stride_h - pad_h + r * dilation_h;
//             const int w = ow * stride_w - pad_w + s * dilation_w;
//             const int k = oc;
//             if (h < 0 || h >= H || w < 0 || w >= W) continue;
//             sum += I[((n * C + c) * H + h) * W + w] *
//                     F[((k * C + c) * R + r) * S + s];
//           }
//         }
//       }
//       O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
//     }
//   }
// }

__global__ void convolution_kernel_2(float *I, float *F, float *O, int N, int C, int H, int W, int K, int R, int S, int ON, int OC, int OH, int OW, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  // 인접한 메모리를 읽어오기 위해 1차원으로 변환
  int ow = tidx % OW;
  int oh = (tidx / OW) % OH;
  int oc = (tidx / (OW * OH)) % OC;
  int on = (tidx / (OW * OH * OC)) % ON;
  if (tidx >= ON * OC * OH * OW) return;
  float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w) {
  // // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
  //                       stride_w, dilation_h, dilation_w);

  CHECK_CUDA(cudaMemcpy(I_gpu, _I, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_gpu, _F, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice));

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  // dim3 blockDim_1(32, 32);
  // dim3 gridDim_1((ON + blockDim_1.x - 1) / 32, (OC + blockDim_1.y - 1) / 32);
  // convolution_kernel_1<<<gridDim_1, blockDim_1>>>(I_gpu, F_gpu, O_gpu, N, C, H, W, K, R, S, ON, OC, OH, OW, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  dim3 blockDim_2(1024);
  dim3 gridDim_2((ON * OC * OH * OW + 1023) / 1024);
  convolution_kernel_2<<<gridDim_2, blockDim_2>>>(I_gpu, F_gpu, O_gpu, N, C, H, W, K, R, S, ON, OC, OH, OW, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

  CHECK_CUDA(cudaMemcpy(_O, O_gpu, sizeof(float) * ON * OC * OH * OW, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMalloc(&I_gpu, sizeof(float) * N * C * H * W));
  CHECK_CUDA(cudaMalloc(&F_gpu, sizeof(float) * K * C * R * S));
  CHECK_CUDA(cudaMalloc(&O_gpu, sizeof(float) * ON * OC * OH * OW));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {

  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}