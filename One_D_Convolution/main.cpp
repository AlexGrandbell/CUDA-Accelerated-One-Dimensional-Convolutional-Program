//纯CPU程序
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 定义大小
#define KERNEL_SIZE 30000
#define INPUT_SIZE 1000000

// 一维卷积函数
void convolution(float* input, float* kernel, float* output) {
    for (int i = 0; i < INPUT_SIZE - KERNEL_SIZE + 1; ++i) {
        output[i] = 0.0f;
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            output[i] += input[i + j] * kernel[j];
        }
    }
}

int main() {
    // 计时
    clock_t start_time = clock();
    // 随机生成输入序列
    srand((unsigned int)time(NULL));

    //随机生成卷积x
    float input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; ++i) {
        input[i] = (float)rand() / RAND_MAX;//生成范围在0-1之间的元素
    }

    // 随机生成卷积核
    const int kernel_size = KERNEL_SIZE;
    float kernel[kernel_size];
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] = (float)rand() / RAND_MAX;
    }

    // 计算输出序列
    const int output_size = INPUT_SIZE - kernel_size + 1;
    float output[output_size];

//    // 计时
//    clock_t start_time = clock();

    // 执行卷积运算
    convolution(input, kernel, output);

    // 计时结束
    clock_t end_time = clock();
    double cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // 打印运行时间
    printf("CPU运行时间: %f 秒\n", cpu_time_used);

    return 0;
}
