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

	h_ged.exclPar = (float2*)calloc(atomTypesCount*atomTypesCount, sizeof(float2));
	cudaMalloc((void**)&d_ged.exclPar, atomTypesCount*atomTypesCount*sizeof(float2));
	h_ged.gaussCount = (int*)calloc(atomTypesCount*atomTypesCount, sizeof(int));
	cudaMalloc((void**)&d_ged.gaussCount, atomTypesCount*atomTypesCount*sizeof(int));

	maxGaussCount = 0;
	int i, j, k;
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			h_ged.exclPar[i*atomTypesCount + j].x = (float)gauss[i+j*atomTypesCount].l;
			h_ged.exclPar[i*atomTypesCount + j].y = gauss[i+j*atomTypesCount].A;
			h_ged.gaussCount[i*atomTypesCount + j] = gauss[i+j*atomTypesCount].numberGaussians;
			if(gauss[i+j*atomTypesCount].numberGaussians > maxGaussCount){
				maxGaussCount = gauss[i+j*atomTypesCount].numberGaussians;
			}
		}
	}

	h_ged.gaussPar = (float3*)calloc(atomTypesCount*atomTypesCount*maxGaussCount, sizeof(float3));
	cudaMalloc((void**)&d_ged.gaussPar, atomTypesCount*atomTypesCount*maxGaussCount*sizeof(float3));

	pdisp = atomTypesCount*atomTypesCount;

	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			for(k = 0; k < gauss[i+j*atomTypesCount].numberGaussians; k++){
				h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].x = gauss[i+j*atomTypesCount].B[k];
				h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].y = gauss[i+j*atomTypesCount].C[k];
				h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].z = gauss[i+j*atomTypesCount].R[k];
			}
		}
	}

/*	printf("Excluded volume parameters:\n");
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			printf("(%5.4e, %5.4e)\t", h_ged.exclPar[i*atomTypesCount + j].x, h_ged.exclPar[i*atomTypesCount + j].y);
		}
		printf("\n");
	}

	printf("Gauss count:\n");
	for(i = 0; i < atomTypesCount; i++){
		for(j = 0; j < atomTypesCount; j++){
			printf("%d\t", h_ged.gaussCount[i*atomTypesCount + j]);
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
						h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].x);
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
						h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].y);
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
						h_ged.gaussPar[k*pdisp + i*atomTypesCount + j].z);
			}
			printf("\n");
		}
	}
	exit(0);*/
	h_ged.energies = (float2*)calloc(mdd->N, sizeof(float2));
	cudaMalloc((void**)&d_ged.energies, mdd->N*sizeof(float2));

	cudaMemcpy(d_ged.exclPar, h_ged.exclPar, atomTypesCount*atomTypesCount*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ged.gaussCount, h_ged.gaussCount, atomTypesCount*atomTypesCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ged.gaussPar, h_ged.gaussPar, atomTypesCount*atomTypesCount*maxGaussCount*sizeof(float3), cudaMemcpyHostToDevice);

	printf("Done initializing GaussExcluded potential\n");
}

GaussExcluded::~GaussExcluded(){

}

__global__ void gaussExcluded_kernel(int* d_pairsCount, int* d_pairsList, GEData d_ged, int widthTot, int atomTypesCount, float cutoff){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
//	int patom = 3740;
	if(d_i < c_mdd.N){
		int p,j,k;
		int pdisp = atomTypesCount*atomTypesCount;
		float4 f = c_mdd.d_force[d_i];
//		float4 f = make_float4(0.0, 0.0, 0.0, 0.0);
		float4 r1 = tex1Dfetch(t_coord, d_i);
		int aty1 = tex1Dfetch(t_atomTypes, d_i);
		float4 r2;
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*widthTot + d_i];
			int aty2 = tex1Dfetch(t_atomTypes, j);
			int ij = aty2*atomTypesCount + aty1;
			r2 = tex1Dfetch(t_coord, j);
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float r = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

			if(r < cutoff){

				float2 exclPar = d_ged.exclPar[ij];

//				if (exclPar.y>0) {
				float mult = exclPar.y*exclPar.x/powf(r, exclPar.x+2.0f);

/*				if (d_i==patom) {
					float4 r3 = tex1Dfetch(t_coord, j);
					printf("d_i: %d j: %d aty1: %d aty2: %d ij: %d\n", d_i, j, aty1, aty2, ij);
					printf("x(i): %f %f %f x(j): %f %f %f\n", r1.x, r1.y, r1.z, r3.x, r3.y, r3.z);
					printf("r: %f dx: %f %f %f\n",r, r2.x, r2.y, r2.z);
					printf("EXCLUDED A: %f L: %f mult: %f\n", exclPar.y, exclPar.x, mult);
				}*/

				for (k=0;k<d_ged.gaussCount[ij];++k) {
					float3 gaussPar = d_ged.gaussPar[pdisp*k + ij];
					float dr = (r-gaussPar.z);
					mult += 2.0f*gaussPar.y*gaussPar.x*expf(-gaussPar.y*dr*dr)*dr/r;
					//float gmult = 2.0f*gaussPar.y*gaussPar.x*expf(-gaussPar.y*dr*dr)*dr/r;
					//if (d_i==patom) printf("GAUSS k: %d B: %f C: %f R: %f dr: %f mult: %f\n",k, gaussPar.x, gaussPar.y, gaussPar.z, dr, gmult);
				}

				//if (d_i==patom) printf("FORCES mult %f f: %f %f %f\n\n", mult, mult*r2.x, mult*r2.y, mult*r2.z);

				f.x -= mult*r2.x;
				f.y -= mult*r2.y;
				f.z -= mult*r2.z;
//				}
			}
		}
		c_mdd.d_force[d_i] = f;
	}
}

void GaussExcluded::compute(MDData *mdd){
	gaussExcluded_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, d_ged, mdd->widthTot, atomTypesCount, cutoff);
	/*int i;
	cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	FILE* file = fopen("gauss_forces.dat", "w");
	for(i = 0; i < mdd->N; i++){
		fprintf(file, "%f %f %f\n", mdd->h_force[i].x, mdd->h_force[i].y, mdd->h_force[i].z);
	}
	fclose(file);
	exit(0);*/
}

__global__ void gaussExcludedEnergy_kernel(int* d_pairsCount, int* d_pairsList, GEData d_ged, int atomTypesCount, float cutoff){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		int p,j,k;
		int pdisp = atomTypesCount*atomTypesCount;
		float4 r1 = tex1Dfetch(t_coord, d_i);
		int aty1 = tex1Dfetch(t_atomTypes, d_i);
		float4 r2;
		float2 energies = make_float2(0.0f, 0.0f);
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*c_mdd.widthTot + d_i];
			int aty2 = tex1Dfetch(t_atomTypes, j);
			int ij = aty2*atomTypesCount + aty1;
			r2 = tex1Dfetch(t_coord, j);
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float r = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

			if(r < cutoff){

				float2 exclPar = d_ged.exclPar[ij];

//				if (exclPar.y>0) {
				energies.y += exclPar.y/powf(r, exclPar.x);

				for (k=0;k<d_ged.gaussCount[ij];++k) {
					float3 gaussPar = d_ged.gaussPar[pdisp*k + ij];
					float dr = (r-gaussPar.z);
					energies.x += gaussPar.x*expf(-gaussPar.y*dr*dr);
				}

//				}
			}
		}
		d_ged.energies[d_i] = energies;
	}
}

float GaussExcluded::get_energies(int energy_id, int timestep){
	if(timestep != lastStepEnergyComputed){
		gaussExcludedEnergy_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, d_ged, atomTypesCount, cutoff);
		cudaMemcpy(h_ged.energies, d_ged.energies, mdd->N*sizeof(float2), cudaMemcpyDeviceToHost);
		energyValues[0] = 0.0f;
		energyValues[1] = 0.0f;
		int i;
		for(i = 0; i < mdd->N; i++){
			energyValues[0] += h_ged.energies[i].x;
			energyValues[1] += h_ged.energies[i].y;
		}
		energyValues[0] /= 2.0f;
		energyValues[1] /= 2.0f;
		lastStepEnergyComputed = timestep;
	}
	return energyValues[energy_id];
}

