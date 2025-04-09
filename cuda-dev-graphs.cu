//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;

#define ARRAY_SIZE (1 << 25)
#define ARRAY_SIZE_IN_BYTES (sizeof(int) * (ARRAY_SIZE))
#define NUM_STREAMS 8

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void work_kernel(int *a, int* result, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = tid; i < N; i += stride) {
		result[i] = a[i] + 100;
	}
}


int main(int argc, char** argv)
{
	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block
	int N = 10;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate and initialize memory
	int* a_cpu;
	int* result_cpu;
	a_cpu = (int*)malloc(N * sizeof(int));
	result_cpu = (int*)malloc(N * sizeof(int));

	for (int i = 0; i < N; i++) {
		a_cpu[i] = 1;
	}

	int* a_gpu;
	int* result_gpu;
	checkCuda( cudaMalloc((void **)&a_gpu, N*sizeof(int)) );
	checkCuda( cudaMalloc((void **)&result_gpu, N*sizeof(int)) );
	cudaMemcpy( a_gpu, a_cpu, N*sizeof(int), cudaMemcpyHostToDevice );

	cudaEventRecord(start);
	work_kernel<<<1, 1>>>(a_gpu, result_gpu, N);
	cudaEventRecord(stop);

	// Copy back results, synchronize
	checkCuda( cudaMemcpy( result_cpu, result_gpu, N*sizeof(int), cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	if (true) {
		printf("Results of operation: \n");
		for (int i = 0; i < N; i++) {
			printf("Result[%d]: %d\n", i, result_cpu[i]);
		}
	}

	// Print runtime information
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds elapsed: %f\n", milliseconds);

	// Free memory
	checkCuda( cudaFree(a_gpu) );
	checkCuda( cudaFree(result_gpu) );
	free(a_cpu);
	free(result_cpu);
}
