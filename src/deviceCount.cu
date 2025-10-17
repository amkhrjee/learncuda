#include <iostream>

__global__ void deviceInfo(int *device)
{
    cudaGetDevice(device);
}

int main(void)
{
    int count;
    int *device;
    int h_device;

    cudaGetDeviceCount(&count);

    cudaMalloc(&device, sizeof(int));

    std::cout << "Device count: " << count << std::endl;

    deviceInfo<<<1, 1>>>(device);

    cudaMemcpy(&h_device, device, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Device: " << h_device << std::endl;

    return 0;
}