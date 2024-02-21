#include <cstdio>
#include <time.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void pythagoras(int *pa, int *pb, int *pc, int *presult) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *presult = 1;
  else
    *presult = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int result = 0;
  clock_t time1, time2, time3, time4, time5, time11, time12, time13, time14, time15;
  time1 = clock();
  // TODO: 1. allocate device memory
  int *d_a, *d_b, *d_c, *d_result;
  d_a = d_b = d_c = d_result = nullptr;
  time11 = clock();
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(int)));
  time12 = clock();
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(int)));
  time13 = clock();
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));
  time14 = clock();
  CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
  time15 = clock();

  time2 = clock();
  // TODO: 2. copy data to device
  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

  time3 = clock();
  // TODO: 3. launch kernel
  pythagoras<<<1, 1>>>(d_a, d_b, d_c, d_result);

  time4 = clock();
  // TODO: 4. copy result back to host
  cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  time5 = clock();
  if (result) printf("YES\n");
  else printf("NO\n");
  
  printf("Time for TODO 1: %f\n", (double)(time2 - time1) / CLOCKS_PER_SEC);
  printf("Time for TODO 2: %f\n", (double)(time3 - time2) / CLOCKS_PER_SEC);
  printf("Time for TODO 3: %f\n", (double)(time4 - time3) / CLOCKS_PER_SEC);
  printf("Time for TODO 4: %f\n", (double)(time5 - time4) / CLOCKS_PER_SEC);

  printf("Time for cudaMalloc d_a: %f\n", (double)(time12 - time11) / CLOCKS_PER_SEC);
  printf("Time for cudaMalloc d_b: %f\n", (double)(time13 - time12) / CLOCKS_PER_SEC);
  printf("Time for cudaMalloc d_c: %f\n", (double)(time14 - time13) / CLOCKS_PER_SEC);
  printf("Time for cudaMalloc d_result: %f\n", (double)(time15 - time14) / CLOCKS_PER_SEC);

  return 0;
}
