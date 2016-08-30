/*
 * PairlistUpdater.cu
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#include "PairlistUpdater.cuh"
#include "ComputationalArrays.h"
#include "../Util/ReductionAlgorithms.cuh"

__global__ void countL1Pairs_kernel(int* d_exclusionsCount, int* d_exclusionsList, int* d_pairsCount, float cutoffSq, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 ri = tex1Dfetch(t_coord, d_i);
		int count = 0;
		int j;
		int currentExl = d_exclusionsList[d_i];
		int currentExclIdx = 0;
		for(j = 0; j < N; j++){
			if(j != currentExl){
				if(j != d_i){
					float4 rj = tex1Dfetch(t_coord, j);
					rj.x -= ri.x;
					rj.y -= ri.y;
					rj.z -= ri.z;

					float3 pb = c_mdd.bc.len;
					rj.x -= rint(rj.x/pb.x)*pb.x;
					rj.y -= rint(rj.y/pb.y)*pb.y;
					rj.z -= rint(rj.z/pb.z)*pb.z;

					rj.w = rj.x*rj.x + rj.y*rj.y + rj.z*rj.z;
					if(rj.w < cutoffSq){
						count++;
					}
				}
			} else {
				currentExclIdx++;
				currentExl = d_exclusionsList[c_mdd.widthTot*currentExclIdx + d_i];
			}
		}
		d_pairsCount[d_i] = count;
	}
}

__global__ void updateL1Pairs_kernel(int* d_exclusionsCount, int* d_exclusionsList, int* d_pairsCount, int* d_pairsList,
		float4* d_old_coord, float cutoffSq, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 ri = tex1Dfetch(t_coord, d_i);
		int count = 0;
		int j;
		int currentExl = d_exclusionsList[d_i];
		int currentExclIdx = 0;
		for(j = 0; j < N; j++){
			if(j != currentExl){
				if(j != d_i){
					float4 rj = tex1Dfetch(t_coord, j);
					rj.x -= ri.x;
					rj.y -= ri.y;
					rj.z -= ri.z;

					float3 pb = c_mdd.bc.len;
					rj.x -= rint(rj.x/pb.x)*pb.x;
					rj.y -= rint(rj.y/pb.y)*pb.y;
					rj.z -= rint(rj.z/pb.z)*pb.z;

					rj.w = rj.x*rj.x + rj.y*rj.y + rj.z*rj.z;
					if(rj.w < cutoffSq){
						d_pairsList[c_mdd.widthTot*count + d_i] = j;
						count++;
					}
				}
			} else {
				currentExclIdx++;
				currentExl = d_exclusionsList[c_mdd.widthTot*currentExclIdx + d_i];
			}
		}
		d_pairsCount[d_i] = count;
		d_old_coord[d_i] = ri;
	}
}

__global__ void countL2Pairs_kernel(int* d_pairsL1Count, int* d_pairsL1List, int* d_pairsCount, float cutoffSq, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 ri = tex1Dfetch(t_coord, d_i);
		int count = 0;
		int j, pj;
		for(pj = 0; pj < d_pairsL1Count[d_i]; pj++){
			j = d_pairsL1List[pj*c_mdd.widthTot + d_i];
			float4 rj = tex1Dfetch(t_coord, j);
			rj.x -= ri.x;
			rj.y -= ri.y;
			rj.z -= ri.z;

			float3 pb = c_mdd.bc.len;
			rj.x -= rint(rj.x/pb.x)*pb.x;
			rj.y -= rint(rj.y/pb.y)*pb.y;
			rj.z -= rint(rj.z/pb.z)*pb.z;

			rj.w = rj.x*rj.x + rj.y*rj.y + rj.z*rj.z;
			if(rj.w < cutoffSq){
				count++;
			}
		}
		d_pairsCount[d_i] = count;
	}
}

__global__ void updateL2Pairlist_kernel(int* d_pairsL1Count, int* d_pairsL1List, int* d_pairsCount, int* d_pairsList,
		float4* d_old_coord, float cutoffSq, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 ri = tex1Dfetch(t_coord, d_i);
		int count = 0;
		int j, pj;
		for(pj = 0; pj < d_pairsL1Count[d_i]; pj++){
			j = d_pairsL1List[pj*c_mdd.widthTot + d_i];
			float4 rj = tex1Dfetch(t_coord, j);
			rj.x -= ri.x;
			rj.y -= ri.y;
			rj.z -= ri.z;

			float3 pb = c_mdd.bc.len;
			rj.x -= rint(rj.x/pb.x)*pb.x;
			rj.y -= rint(rj.y/pb.y)*pb.y;
			rj.z -= rint(rj.z/pb.z)*pb.z;

			rj.w = rj.x*rj.x + rj.y*rj.y + rj.z*rj.z;
			if(rj.w < cutoffSq){
				d_pairsList[count*c_mdd.widthTot + d_i] = j;
				count++;
			}
		}
		d_pairsCount[d_i] = count;
		d_old_coord[d_i] = ri;
	}
}


PairlistUpdater::PairlistUpdater(MDData *mdd) {

	blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	cudaMalloc( (void **) &d_drsq, mdd->N * sizeof(float));
	cudaMalloc( (void **) &d_old_coord, mdd->N * sizeof(float4));

	reduction = new Reduction(mdd->N);

	cudaMemcpy(d_old_coord, mdd->d_coord, mdd->N*sizeof(float4), cudaMemcpyDeviceToDevice);

	h_pairs.count = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_pairs.count, mdd->N*sizeof(int));

	timesUpdated = 0;
}

PairlistUpdater::~PairlistUpdater(){
	free(h_pairs.count);
	cudaFree(d_pairs.count);
	cudaFree(d_drsq);
	cudaFree(d_old_coord);
}

void PairlistUpdater::update(MDData *mdd){
	float drsq_max = rmax_displacement(d_old_coord, mdd->d_coord, mdd->N);
	if(drsq_max > drsq_up){
		if(drsq_max > 4.0f*drsq_up){
			printf("WARNING: Displacement of at least one particle is larger (%f) then the difference between pairlist cutoff and non-bonded potential cutoff.\n", drsq_max);
		}
		this->doUpdate(mdd);
		timesUpdated ++;
	}
}

__global__ void calc_displacement_kernal(float4* d_r1, float4* d_r2, float *d_drsq, int N)
{
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < N){
		float4 r1 = d_r1[d_i];
		float4 r2 = d_r2[d_i];
		r2.x -= r1.x;
		r2.y -= r1.y;
		r2.z -= r1.z;

		float3 pb = c_mdd.bc.len;
		r2.x -= rint(r2.x/pb.x)*pb.x;
		r2.y -= rint(r2.y/pb.y)*pb.y;
		r2.z -= rint(r2.z/pb.z)*pb.z;

		r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
		d_drsq[d_i] = r2.w;
	}
}

float PairlistUpdater::rmax_displacement(float4* d_r1, float4* d_r2, int N)
{
	calc_displacement_kernal<<<this->blockCount, this->blockSize>>>(d_r1, d_r2, d_drsq, N);

	return reduction->rmax(d_drsq);
}

void PairlistUpdater::printPairlist(){
	cudaMemcpy(h_pairs.count, d_pairs.count, mdd->N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pairs.list, d_pairs.list, mdd->N*sizeof(int)*maxPairsPerAtom, cudaMemcpyDeviceToHost);
	for(int i = 0; i < mdd->N; i ++){
		printf("%d: %d\t", i, h_pairs.count[i]);
		for(int j = 0; j < h_pairs.count[i]; j++){
			printf("%d\t", h_pairs.list[j*mdd->widthTot + i]);
		}
		printf("\n");
	}
}
