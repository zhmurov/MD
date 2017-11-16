/*
 * ReductionAlgorithmsFloat4.cu
 *
 *  Created on: 02.04.2016
 *      Author: zhmurov
 */

#include "ReductionAlgorithmsFloat4.cuh"
#include <float.h>

#define BLOCK_SIZE	64

ReductionFloat4::ReductionFloat4(int nn)
{
	n = nn;

	blockSize = BLOCK_SIZE;

	int numBlocks = (n-1)/BLOCK_SIZE+1;
	int numBlocks2 = (numBlocks-1)/BLOCK_SIZE+1;

	cudaMalloc( (void **) &d_array, numBlocks * sizeof(float4) );
	cudaMalloc( (void **) &d_array2, numBlocks2 * sizeof(float4) );
	h_array = new float4[BLOCK_SIZE];
}

ReductionFloat4::~ReductionFloat4()
{
	delete [] h_array;

	cudaFree (d_array);
	cudaFree (d_array2);
}

float4 ReductionFloat4::rsum(float4 *d_a)
{
	float4 sum = make_float4(0.0, 0.0, 0.0, 0.0);

	call_reduction_kernal(d_a, reduce_sum_float4_kernal);

	for (int i=0;i<numBlocks;i++) {
		sum.x += h_array[i].x;
		sum.y += h_array[i].y;
		sum.z += h_array[i].z;
		sum.w += h_array[i].w;
	}

	return sum;
}

inline void ReductionFloat4::call_reduction_kernal(float4 *d_a, void (*f_kernal)(float4 *, float4 *, int) )
{
	int m;

	numBlocks = (n-1)/(2*BLOCK_SIZE)+1;

	(*f_kernal)<<<numBlocks, blockSize>>>(d_a, d_array, n);

	p1 = d_array;
	p2 = d_array2;

	int i = 0;
	while (numBlocks>blockSize) {
		m = numBlocks;
		numBlocks = (numBlocks-1)/(2*BLOCK_SIZE)+1;

		(*f_kernal)<<<numBlocks, blockSize>>>(p1, p2, m);

		p3 = p1; p1 = p2; p2 = p3;
		i++;
	}

	cudaMemcpy(h_array, p1, numBlocks * sizeof(float4), cudaMemcpyDeviceToHost);
}

__global__ void reduce_sum_float4_kernal(float4 *in_a, float4 *out_a, int m)
{
	__shared__ float4 a[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;

	a[tid] = make_float4(0.0, 0.0, 0.0, 0.0);
	if(i+blockDim.x < m) {
		float4 in1 = in_a[i];
		float4 in2 = in_a[i+blockDim.x];
		a[tid].x = in1.x + in2.x;
		a[tid].y = in1.y + in2.y;
		a[tid].z = in1.z + in2.z;
		a[tid].w = in1.w + in2.w;
	} else
	if(i<m){
		a[tid] = in_a[i];
	}

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			a[tid].x += a[tid + s].x;
			a[tid].y += a[tid + s].y;
			a[tid].z += a[tid + s].z;
			a[tid].w += a[tid + s].w;
		}
		__syncthreads();
	}

	if (tid==0) out_a[blockIdx.x] = a[0];
}
