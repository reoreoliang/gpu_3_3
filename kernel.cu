#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>

// 以下代码是为了演示共享内存和线程同步的使用
// 其效果是计算均值
__global__ void gpu_shared_memory(float* d_a) {
	int i, index = threadIdx.x;  // 索引从0开始
	float average, sum = 0.0f;
	// 定义共享内存
	__shared__ float sh_arr[10];
	sh_arr[index] = d_a[index];  // 将数据从全局内存写入到共享内存中

	// 下面这个指令的作用是确保在继续执行程序之前先完成对内存的所有写入操作
	__syncthreads();
	for (i = 0; i <= index; i++)  // 并行循环
	{
		sum += sh_arr[i];
	}
	average = sum / (index + 1.0f);
	d_a[index] = average;

	sh_arr[index] = average;  // 这里只是为了演示一下共享内存的生存周期，注意译者注
}

int main(void)
{
	float h_a[10], * d_a;
	// 数值赋值
	for (int i = 0; i < 10; i++)
	{
		h_a[i] = i;
	}

	cudaMalloc((void **)&d_a, 10 * sizeof(float));
	cudaMemcpy((void *)d_a, (void *)h_a, 10 * sizeof(float), cudaMemcpyHostToDevice);
	gpu_shared_memory << <1, 10 >> > (d_a);
	cudaMemcpy((void*)h_a, (void*)d_a, 10 * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Use of shared Memory on GPU\n");
	for (int i = 0; i < 10; i++)
	{
		printf("The running average after %d element is %f \n", i, h_a[i]);
	}
	return 0;
}