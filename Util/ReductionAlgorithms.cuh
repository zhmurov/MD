/*
 * ReductionAlgorithms.cuh
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 */

#pragma once

class Reduction
{
public:
	Reduction(int nn);
	~Reduction();

	float rsum(float *d_a);
	float rmax(float *d_a);
	float rmin(float *d_a);

private:
	int n;
	float *d_array, *d_array2, *p1, *p2, *p3;
	float *h_array;
	int numBlocks, blockSize;

	inline void call_reduction_kernal(float *d_a, void (*f_kernal)(float *, float *, int));
};

__global__ void reduce_sum_kernal(float *in_a, float *out_a, int m);
__global__ void reduce_max_kernal(float *in_a, float *out_a, int m);
__global__ void reduce_min_kernal(float *in_a, float *out_a, int m);
