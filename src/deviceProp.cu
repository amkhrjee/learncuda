#include <iostream>

int main(void)
{
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties_v2(&prop, i);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Device clock rate: " << prop.clockRate / 1e3 << " MHz" << std::endl;
        std::cout << "Device ECC enabled: " << (prop.ECCEnabled ? "True" : "False") << std::endl;
        std::cout << "Device Max Grid Support: "
                  << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
        std::cout << "Device Warp size: " << prop.warpSize << std::endl;
        std::cout << "Device Sparse Array support: " << (prop.sparseCudaArraySupported ? "True" : "False") << std::endl;
        std::cout << "Device Streaming Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Device Memory: " << prop.totalGlobalMem / (1 << 30) << "GB" << std::endl;
        std::cout << "Device Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Device Total Constant Memory: " << prop.totalConstMem / (1 << 10) << "KB"  << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
}
