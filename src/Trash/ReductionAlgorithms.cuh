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

	float rsum(float *a);
	float rmax(float *a, int n);
	float rmin(float *a, int n);

private:
	int n;
	float *d_sums, *d_sums2, *p1, *p2, *p3;
	float *h_sums;
};

__global__ void reduce_sum_kernal(float *in_a, float *out_a, int m);
