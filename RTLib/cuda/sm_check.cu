#include <cuda_runtime.h>
#include <iostream>
int main(){
    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp,0);
    std::cout << "arch=" << deviceProp.major << deviceProp.minor << "\n";
    return 0;
}