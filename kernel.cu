#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>

// ���´�����Ϊ����ʾ�����ڴ���߳�ͬ����ʹ��
// ��Ч���Ǽ����ֵ
__global__ void gpu_shared_memory(float* d_a) {
	int i, index = threadIdx.x;  // ������0��ʼ
	float average, sum = 0.0f;
	// ���干���ڴ�
	__shared__ float sh_arr[10];
	sh_arr[index] = d_a[index];  // �����ݴ�ȫ���ڴ�д�뵽�����ڴ���

	// �������ָ���������ȷ���ڼ���ִ�г���֮ǰ����ɶ��ڴ������д�����
	__syncthreads();
	for (i = 0; i <= index; i++)  // ����ѭ��
	{
		sum += sh_arr[i];
	}
	average = sum / (index + 1.0f);
	d_a[index] = average;

	sh_arr[index] = average;  // ����ֻ��Ϊ����ʾһ�¹����ڴ���������ڣ�ע������ע
}

int main(void)
{
	float h_a[10], * d_a;
	// ��ֵ��ֵ
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