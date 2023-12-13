#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <cmath>

#define KERNEL_SIZE 30000
#define INPUT_SIZE 1000000

//初始化卷积序列和卷积核
__global__
void initConvolution(double* input, double* kernel){
    int index = blockIdx.x * blockDim.x + threadIdx.x;//当前下标
    int stride = blockDim.x * gridDim.x;//步长
    curandState state;
    curand_init(1234, index, 0, &state);  // 设置种子，可以根据需要更改

    for (int i = index; i < INPUT_SIZE; i+=stride) {
        input[i] = curand_uniform(&state);  // 生成范围为[0,1)的元素
    }
    for (int i = index; i < KERNEL_SIZE; ++i) {
        kernel[i] = curand_uniform(&state);  // 生成范围为[0,1)的元素
    }
}

//GPU异步卷积操作
__global__
void convolutionForGPU(double* input, double* kernel, double* output,int number,int all) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;//当前下标
    int stride = blockDim.x * gridDim.x;//步长

    for (int i = (INPUT_SIZE - KERNEL_SIZE + 1)*((number-1)/all) + index; i < (INPUT_SIZE - KERNEL_SIZE + 1)*((number)/all) +index ; i+=stride) {
        output[i]=0.0f;
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            output[i]+=input[i+j]*kernel[j];
        }
    }
}

int main(const int argc, const char** argv) {
    int numberOfSMs,deviceId;
    size_t numberOfBlocks,threadsPerBlock;//线程块数和块中线程数
    cudaGetDevice(&deviceId);//获取GPU的ID
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    threadsPerBlock = 256;//设置线程数
    numberOfBlocks = 32 * numberOfSMs;//设置线程块数

    //使用异步流进一步加速
    cudaStream_t stream1,stream2,stream3,stream4,stream5,stream6;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);


    //输入序列、卷积核序列、输出序列与分配内存
    double* input, * kernel, * outputGPU;
    size_t INPUT_SIZE_t,KERNEL_SIZE_t,output_size_t;
    INPUT_SIZE_t = INPUT_SIZE * sizeof(double);
    KERNEL_SIZE_t = KERNEL_SIZE * sizeof(double);
    output_size_t = (INPUT_SIZE - KERNEL_SIZE + 1) * sizeof(double);
    cudaMallocManaged(&input, INPUT_SIZE_t);
    cudaMallocManaged(&kernel, KERNEL_SIZE_t);
    cudaMallocManaged(&outputGPU, output_size_t);

    //内存预取
    cudaMemPrefetchAsync(input,INPUT_SIZE_t,deviceId);
    cudaMemPrefetchAsync(kernel,KERNEL_SIZE_t,deviceId);
    cudaMemPrefetchAsync(outputGPU,output_size_t,deviceId);

    //开始时间
    clock_t start_time = clock();

    //GPU初始化元素
    initConvolution<<<numberOfBlocks,threadsPerBlock>>>(input,kernel);
    cudaDeviceSynchronize();
    //GPU处理卷积
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream1>>>(input, kernel, outputGPU,1,6);
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream2>>>(input, kernel, outputGPU,2,6);
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream3>>>(input, kernel, outputGPU,3,6);
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream4>>>(input, kernel, outputGPU,4,6);
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream5>>>(input, kernel, outputGPU,5,6);
    convolutionForGPU<<<numberOfBlocks, threadsPerBlock,0,stream6>>>(input, kernel, outputGPU,6,6);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
    cudaStreamSynchronize(stream5);
    cudaStreamSynchronize(stream6);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);

    clock_t end_time = clock();
    double gpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("运行时间: %f 秒\n", gpu_time_used);

    cudaFree(input);
    cudaFree(kernel);
    cudaFree(outputGPU);

    return 0;
}
