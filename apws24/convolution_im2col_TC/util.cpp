#include "util.h"

#include <omp.h>
#include <sys/time.h>

#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

half *alloc_tensor(int N, int C, int H, int W) {
  half *m;
  CHECK_CUDA(cudaMallocHost(&m, (size_t)N * C * H * W * sizeof(half)));
  return m;
}

float *alloc_tensor32(int N, int C, int H, int W) {
  float *m;
  CHECK_CUDA(cudaMallocHost(&m, (size_t)N * C * H * W * sizeof(float)));
  return m;
}

void rand_tensor(half *m, int N, int C, int H, int W) {
  size_t L = (size_t)N * C * H * W;
  for (size_t j = 0; j < L; j++) { m[j] = (half) ((double)rand() / RAND_MAX - 0.5); }
}

void zero_tensor(half *m, int N, int C, int H, int W) {
  size_t L = (size_t)N * C * H * W;
  memset((void *) m, 0, sizeof(half) * L);
}

void zero_tensor32(float *m, int N, int C, int H, int W) {
  size_t L = (size_t)N * C * H * W;
  memset((void *) m, 0, sizeof(float) * L);
}

void print_tensor(half *m, int N, int C, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      printf("Batch %d, Channel %d\n", n, c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          printf("%+.3f ", (float) (m[(((size_t)n * C + c) * H + h) * W + w]));
        }
        printf("\n");
      }
    }
  }
}

void print_tensor32(float *m, int N, int C, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      printf("Batch %d, Channel %d\n", n, c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          printf("%+.3f ", (float) (m[(((size_t)n * C + c) * H + h) * W + w]));
        }
        printf("\n");
      }
    }
  }
}

void check_convolution(half *I, half *F, float *O, int N, int C, int H, int W,
                       int K, int R, int S, int pad_h, int pad_w, int stride_h,
                       int stride_w, int dilation_h, int dilation_w) {
  float *O_ans;
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  O_ans = alloc_tensor32(ON, OC, OH, OW);
  zero_tensor32(O_ans, ON, OC, OH, OW);

#pragma omp parallel for
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0.0f;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += (float)(I[(((size_t)n * C + c) * H + h) * W + w] *
                       F[(((size_t)k * C + c) * R + r) * S + s]);
              }
            }
          }
          O_ans[(((size_t)on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3f;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = O[(((size_t)on * OC + oc) * OH + oh) * OW + ow];
          float o_ans = O_ans[(((size_t)on * OC + oc) * OH + oh) * OW + ow];
          if (fabs(fabsf(o) - fabs(o_ans)) > eps &&
              (o_ans == 0 || fabs(fabs(fabs(o) - fabs(o_ans)) / o_ans) > eps)) {
            ++cnt;
            if (cnt <= thr)
              printf(
                  "O[%d][%d][%d][%d] : correct_value = %f, your_value = %f\n",
                  on, oc, oh, ow, (float) o_ans, (float) o);
            if (cnt == thr + 1)
              printf("Too many error, only first %d values are printed.\n",
                     thr);
            is_valid = false;
          }
        }
      }
    }
  }

  if (is_valid) {
    printf("Validation Result: VALID\n");
  } else {
    printf("Validation Result: INVALID\n");
  }
}
