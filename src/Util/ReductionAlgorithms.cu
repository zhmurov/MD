/*
 * ReductionAlgorithms.cu
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 */

#include "ReductionAlgorithms.cuh"
#include <float.h>

#define BLOCK_SIZE	64

Reduction::Reduction(int nn)
{
	n = nn;

	blockSize = BLOCK_SIZE;

	int numBlocks = (n-1)/BLOCK_SIZE+1;
	int numBlocks2 = (numBlocks-1)/BLOCK_SIZE+1;

	cudaMalloc( (void **) &d_array, numBlocks * sizeof(float) );
	cudaMalloc( (void **) &d_array2, numBlocks2 * sizeof(float) );
	h_array = new float[BLOCK_SIZE];
}

Reduction::~Reduction()
{
	delete [] h_array;

	cudaFree (d_array);
	cudaFree (d_array2);
}

float Reduction::rsum(float *d_a)
{
	float sum=0.0;

	call_reduction_kernal(d_a, reduce_sum_kernal);

	for (int i=0;i<numBlocks;i++) {
		sum += h_array[i];
	}

	return sum;
}

float Reduction::rmax(float *d_a)
{
	float max = -FLT_MAX;

	call_reduction_kernal(d_a, reduce_max_kernal);

	for (int i=0;i<numBlocks;i++) {
		max = fmaxf(max,h_array[i]);
	}

	return max;
}

float Reduction::rmin(float *d_a)
{
	float min = FLT_MAX;

	call_reduction_kernal(d_a, reduce_min_kernal);

	for (int i=0;i<numBlocks;i++) {
		min = fminf(min,h_array[i]);
	}

	return min;
}

inline void Reduction::call_reduction_kernal(float *d_a, void (*f_kernal)(float *, float *, int) )
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

	cudaMemcpy(h_array, p1, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void reduce_sum_kernal(float *in_a, float *out_a, int m)
{
	__shared__ float a[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;

	a[tid] = 0.0;
	if(i+blockDim.x < m) {
		a[tid] = in_a[i] + in_a[i+blockDim.x];
	} else
	if(i<m){
		a[tid] = in_a[i];
	}

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			a[tid] += a[tid + s];
		}
		__syncthreads();
	}

	if (tid==0) out_a[blockIdx.x] = a[0];
}

__global__ void reduce_max_kernal(float *in_a, float *out_a, int m)
{
	__shared__ float a[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;

	a[tid] = -FLT_MAX;
	if(i+blockDim.x < m) {
		a[tid] = fmaxf(in_a[i], in_a[i+blockDim.x]);
	} else
	if(i<m){
		a[tid] = in_a[i];
	}

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			a[tid] = fmaxf(a[tid], a[tid + s]);
		}
		__syncthreads();
	}

	if (tid==0) out_a[blockIdx.x] = a[0];
}

__global__ void reduce_min_kernal(float *in_a, float *out_a, int m)
{
	__shared__ float a[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;

	a[tid] = FLT_MAX;
	if(i+blockDim.x < m) {
		a[tid] = fminf(in_a[i], in_a[i+blockDim.x]);
	} else if(i<m){
		a[tid] = in_a[i];
	}

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			a[tid] = fminf(a[tid], a[tid + s]);
		}
		__syncthreads();
	}

	if (tid==0) out_a[blockIdx.x] = a[0];
}
