/*
 * ReductionAlgorithms.cu
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 */

#include "ReductionAlgorithms.cuh"

#define BLOCK_SIZE	64

Reduction::Reduction(int nn)
{
	n = nn;

	int numBlocks = (n-1)/BLOCK_SIZE+1;
	int numBlocks2 = (numBlocks-1)/BLOCK_SIZE+1;

	cudaMalloc( (void **) &d_sums, numBlocks * sizeof(float) );
	cudaMalloc( (void **) &d_sums2, numBlocks2 * sizeof(float) );
	h_sums = new float[BLOCK_SIZE];
}

Reduction::~Reduction()
{
	delete [] h_sums;

	cudaFree (d_sums);
	cudaFree (d_sums2);
}

float Reduction::rsum(float *d_a)
{
	float sum=0.0;
	int numBlocks = (n-1)/(2*BLOCK_SIZE)+1;
	int blockSize = BLOCK_SIZE;
	int m;

	reduce_sum_kernal<<<numBlocks, blockSize>>>(d_a, d_sums, n);

	p1 = d_sums;
	p2 = d_sums2;

	int i = 0;
	while (numBlocks>blockSize) {
		m = numBlocks;
		numBlocks = (numBlocks-1)/(2*BLOCK_SIZE)+1;

		reduce_sum_kernal<<<numBlocks, blockSize>>>(p1, p2, m);

		p3 = p1; p1 = p2; p2 = p3;
		i++;
	}

	cudaMemcpy(h_sums, p1, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i=0;i<numBlocks;i++) {
		sum += h_sums[i];
	}

	return sum;
}

/*__global__ void reduce_sum_kernal(float *in_a, float *out_a)
{
	__shared__ float b[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	b[tid] = in_a[i];

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			b[tid] += b[tid + s];
		}
		__syncthreads();
	}

	if (tid==0) out_a[blockIdx.x] = b[0];
}*/

__global__ void reduce_sum_kernal(float *in_a, float *out_a, int m)
{
	__shared__ float a[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;

//	a[tid] = in_a[i] + in_a[i+blockDim.x];

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

float Reduction::rmax(float *a, int n) {return 0.0;}
float Reduction::rmin(float *a, int n) {return 0.0; }
