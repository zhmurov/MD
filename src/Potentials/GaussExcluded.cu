/*
 * GaussExcluded.cu
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */
#include "GaussExcluded.cuh"


GaussExcluded::GaussExcluded(MDData *mdd, float cutoffCONF, int typeCount, GaussExCoeff* gauss, PairlistUpdater *pl){
	printf("Initializing GaussExcluded potential\n");
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->mdd = mdd;

	plist = pl;
	
	cutoff = cutoffCONF;

	lastStepEnergyComputed = -1;

	atomTypesCount = typeCount;

	h_exclPar = (float2*)calloc(atomTypesCount*atomTypesCount, sizeof(float2));
	cudaMalloc((void**)&d_exclPar, atomTypesCount*atomTypesCount*sizeof(float2));
	h_gaussCount = (int*)calloc(atomTypesCount*atomTypesCount, sizeof(int));
	cudaMalloc((void**)&d_gaussCount, atomTypesCount*atomTypesCount*sizeof(int));

	maxGaussCount = 0;
	int i, j, k;
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			h_exclPar[i*atomTypesCount + j].x = (float)gauss[i+j*atomTypesCount].l;
			h_exclPar[i*atomTypesCount + j].y = gauss[i+j*atomTypesCount].A;
			h_gaussCount[i*atomTypesCount + j] = gauss[i+j*atomTypesCount].numberGaussians;
			if(gauss[i+j*atomTypesCount].numberGaussians > maxGaussCount){
				maxGaussCount = gauss[i+j*atomTypesCount].numberGaussians;
			}
		}
	}

	h_gaussPar = (float3*)calloc(atomTypesCount*atomTypesCount*maxGaussCount, sizeof(float3));
	cudaMalloc((void**)&d_gaussPar, atomTypesCount*atomTypesCount*maxGaussCount*sizeof(float3));

	pdisp = atomTypesCount*atomTypesCount;

	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			for(k = 0; k < gauss[i+j*atomTypesCount].numberGaussians; k++){
				h_gaussPar[k*pdisp + i*atomTypesCount + j].x = gauss[i+j*atomTypesCount].B[k];
				h_gaussPar[k*pdisp + i*atomTypesCount + j].y = gauss[i+j*atomTypesCount].C[k];
				h_gaussPar[k*pdisp + i*atomTypesCount + j].z = gauss[i+j*atomTypesCount].R[k];
			}
		}
	}
/*
	printf("Excluded volume parameters:\n");
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			printf("(%5.4e, %5.4e)\t", h_exclPar[i*atomTypesCount + j].x, h_exclPar[i*atomTypesCount + j].y);
		}
		printf("\n");
	}

	printf("Gauss count:\n");
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			printf("%d\t", h_gaussCount[i*atomTypesCount + j]);
		}
		printf("\n");
	}

	printf("=====\nGauss parameters:\n");
	printf("===\nB:\n");
	for(k = 0; k < maxGaussCount; k++){
		printf("k = %d:\n", k+1);
		for(i = 0; i < atomTypesCount; i++){
			for(j = 0; j < atomTypesCount; j++){
				printf("%8.6f\t",
						h_gaussPar[k*pdisp + i*atomTypesCount + j].x);
			}
			printf("\n");
		}
	}
	printf("===\nC:\n");
	for(k = 0; k < maxGaussCount; k++){
		printf("k = %d:\n", k+1);
		for(i = 0; i < atomTypesCount; i++){
			for(j = 0; j < atomTypesCount; j++){
				printf("%8.6f\t",
						h_gaussPar[k*pdisp + i*atomTypesCount + j].y);
			}
			printf("\n");
		}
	}
	printf("===\nR:\n");
	for(k = 0; k < maxGaussCount; k++){
		printf("k = %d:\n", k+1);
		for(i = 0; i < atomTypesCount; i++){
			for(j = 0; j < atomTypesCount; j++){
				printf("%8.6f\t",
						h_gaussPar[k*pdisp + i*atomTypesCount + j].z);
			}
			printf("\n");
		}
	}
	exit(0);
*/
	h_energies = (float2*)calloc(mdd->N, sizeof(float2));
	cudaMalloc((void**)&d_energies, mdd->N*sizeof(float2));

	cudaMemcpy(d_exclPar, h_exclPar, atomTypesCount*atomTypesCount*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussCount, h_gaussCount, atomTypesCount*atomTypesCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gaussPar, h_gaussPar, atomTypesCount*atomTypesCount*maxGaussCount*sizeof(float3), cudaMemcpyHostToDevice);

	printf("Done initializing GaussExcluded potential\n");
}

GaussExcluded::~GaussExcluded(){

}

__global__ void gaussExcluded_kernel(int* d_pairsCount, int* d_pairsList, float2* d_exclPar, int* d_gaussCount, float3* d_gaussPar, int widthTot, int atomTypesCount, float cutoff){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		int p,j,k;
		int pdisp = atomTypesCount*atomTypesCount;
		float4 f = c_mdd.d_force[d_i];
		float4 r1 = c_mdd.d_coord[d_i];
		int aty1 = c_mdd.d_atomTypes[d_i];
		float4 r2;
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*widthTot + d_i];
			int aty2 = c_mdd.d_atomTypes[j];
			int ij = aty2*atomTypesCount + aty1;
			r2 = c_mdd.d_coord[j];
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float r = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

			if(r < cutoff){

				float2 exclPar = d_exclPar[ij];

				float mult = exclPar.y*exclPar.x/powf(r, exclPar.x+2.0f);

				for (k = 0; k < d_gaussCount[ij]; k++) {
					float3 gaussPar = d_gaussPar[pdisp*k + ij];
					float dr = (r-gaussPar.z);
					mult += 2.0f*gaussPar.y*gaussPar.x*expf(-gaussPar.y*dr*dr)*dr/r;
				}

				f.x -= mult*r2.x;
				f.y -= mult*r2.y;
				f.z -= mult*r2.z;
			}
		}
		c_mdd.d_force[d_i] = f;
	}
}

void GaussExcluded::compute(){
	gaussExcluded_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, d_exclPar, d_gaussCount, d_gaussPar, mdd->widthTot, atomTypesCount, cutoff);
}

__global__ void gaussExcludedEnergy_kernel(int* d_pairsCount, int* d_pairsList, float2* d_exclPar, int* d_gaussCount, float3* d_gaussPar, float2* d_energies, int atomTypesCount, float cutoff){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		int p,j,k;
		int pdisp = atomTypesCount*atomTypesCount;
		float4 r1 = c_mdd.d_coord[d_i];
		int aty1 = c_mdd.d_atomTypes[d_i];
		float4 r2;
		float2 energies = make_float2(0.0f, 0.0f);
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*c_mdd.widthTot + d_i];
			int aty2 = c_mdd.d_atomTypes[j];
			int ij = aty2*atomTypesCount + aty1;
			r2 = c_mdd.d_coord[j];
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float r = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

			if(r < cutoff){

				float2 exclPar = d_exclPar[ij];

				energies.y += exclPar.y/powf(r, exclPar.x);

				for (k = 0; k < d_gaussCount[ij]; k++) {
					float3 gaussPar = d_gaussPar[pdisp*k + ij];
					float dr = (r-gaussPar.z);
					energies.x += gaussPar.x*expf(-gaussPar.y*dr*dr);
				}

			}
		}
		d_energies[d_i] = energies;
	}
}

float GaussExcluded::getEnergies(int energyId, int timestep){
	if(timestep != lastStepEnergyComputed){
		gaussExcludedEnergy_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, d_exclPar, d_gaussCount, d_gaussPar, d_energies, atomTypesCount, cutoff);
		cudaMemcpy(h_energies, d_energies, mdd->N*sizeof(float2), cudaMemcpyDeviceToHost);
		energyValues[0] = 0.0f;
		energyValues[1] = 0.0f;
		int i;
		for(i = 0; i < mdd->N; i++){
			energyValues[0] += h_energies[i].x;
			energyValues[1] += h_energies[i].y;
		}
		energyValues[0] /= 2.0f;
		energyValues[1] /= 2.0f;
		lastStepEnergyComputed = timestep;
	}
	return energyValues[energyId];
}

