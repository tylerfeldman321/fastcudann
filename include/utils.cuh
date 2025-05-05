#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>

// Wrapper function to handle CUDA error checking
void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif