/*
 * ReductionAlgorithmsFloat4.cuh
 *
 *  Created on: 02.04.2016
 *      Author: zhmurov
 */

#pragma once

class ReductionFloat4
{
public:
	ReductionFloat4(int nn);
	~ReductionFloat4();

	float4 rsum(float4 *d_a);

private:
	int n;
	float4 *d_array, *d_array2, *p1, *p2, *p3;
	float4 *h_array;
	int numBlocks, blockSize;

	inline void call_reduction_kernal(float4 *d_a, void (*f_kernal)(float4 *, float4 *, int));
};

__global__ void reduce_sum_float4_kernal(float4 *in_a, float4 *out_a, int m);
