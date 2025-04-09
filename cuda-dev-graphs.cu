//Based on the work of Andrew Krepps
#include <stdio.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
using namespace std;

#define ARRAY_SIZE (1 << 25)
#define NSTEP 1000
#define NKERNEL 20


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

void run_with_task_graph(int* a_gpu, int* result_gpu, int N) {
	bool graphCreated=false;
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaGraph_t graph;
	cudaGraphExec_t instance;
	for(int istep=0; istep<10; istep++){
		if(!graphCreated){
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
			for(int ikrnl=0; ikrnl<100; ikrnl++){
				work_kernel<<<256, 256, 0, stream>>>(a_gpu, result_gpu, N);
			}
			cudaStreamEndCapture(stream, &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			graphCreated=true;
		}
		cudaGraphLaunch(instance, stream);
		cudaStreamSynchronize(stream);
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}


float test_task_graph(int* a_gpu, int* result_gpu, int* result_cpu, int N) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	run_with_task_graph(a_gpu, result_gpu, N);
	cudaEventRecord(stop);

	// Copy back results, synchronize
	checkCuda( cudaMemcpy( result_cpu, result_gpu, N*sizeof(int), cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	if (true) {
		printf("[GRAPH] Results of operation: \n");
		for (int i = 0; i < 5; i++) {
			printf("[GRAPH] Result[%d]: %d\n", i, result_cpu[i]);
		}
	}

	// Print runtime information
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[GRAPH] Milliseconds elapsed: %f\n", milliseconds);
	return milliseconds;
}


void run_with_kernel_launches(int* a_gpu, int* result_gpu, int N) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	for(int istep=0; istep<NSTEP; istep++) {
		for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
			work_kernel<<<256, 256, 0, stream>>>(a_gpu, result_gpu, N);
		}
		cudaStreamSynchronize(stream);
	}
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}


float test_kernel_launch(int* a_gpu, int* result_gpu, int* result_cpu, int N) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	run_with_kernel_launches(a_gpu, result_gpu, N);
	cudaEventRecord(stop);

	// Copy back results, synchronize
	checkCuda( cudaMemcpy( result_cpu, result_gpu, N*sizeof(int), cudaMemcpyDeviceToHost ) );
	checkCuda( cudaDeviceSynchronize() );
	if (true) {
		printf("[KERNELLAUNCH] Results of operation: \n");
		for (int i = 0; i < 5; i++) {
			printf("[KERNELLAUNCH] Result[%d]: %d\n", i, result_cpu[i]);
		}
	}

	// Print runtime information
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[KERNELLAUNCH] Milliseconds elapsed: %f\n", milliseconds);
	return milliseconds;
}


int main(int argc, char** argv)
{
	int totalThreads = (1 << 20);
	int blockSize = 256;  // Also threads / block
	int N = (1 << 10);

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

	test_task_graph(a_gpu, result_gpu, result_cpu, N);
	test_kernel_launch(a_gpu, result_gpu, result_cpu, N);

	// Free memory
	checkCuda( cudaFree(a_gpu) );
	checkCuda( cudaFree(result_gpu) );
	free(a_cpu);
	free(result_cpu);
}
