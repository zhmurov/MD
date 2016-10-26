/*
 * Coulomb.cu
 *
 *  Created on: 05.09.2012
 *      Author: zhmurov
 */

#include "Coulomb.cuh"

Coulomb::Coulomb(MDData *mdd, PairlistUpdater *pl, float alpha, float dielectric, float cutoff){
	printf("Initializing Coulomb potential\n");
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	this->mdd = mdd;
	this->plist = pl;

	this->alpha = alpha;
	this->dielectric = dielectric;
	this->cutoff = cutoff;
	this->cutoffSq = cutoff*cutoff;

	kc = QQR2E/dielectric;

	lastStepEnergyComputed = -1;

	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	printf("Done initializing Coulomb potential\n");
}

Coulomb::~Coulomb(){
	free(h_energy);
	cudaFree(d_energy);
}

__global__ void coulomb_kernel(int* d_pairsCount, int* d_pairsList, float kc, float cutoff, int widthTot){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 r1 = tex1Dfetch(t_coord, d_i);
		float q1 = tex1Dfetch(t_charges, d_i);
		float4 r2;
		float4 f = c_mdd.d_force[d_i];
		int p, j;
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*widthTot + d_i];
			r2 = tex1Dfetch(t_coord, j);

			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;
			float q2 = tex1Dfetch(t_charges, j);

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float rsq = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
			float r = sqrtf(rsq);

			if(r < cutoff){
				float mult = -kc*q1*q2/(r*rsq);
				f.x += mult*r2.x;
				f.y += mult*r2.y;
				f.z += mult*r2.z;
			}
		}
		c_mdd.d_force[d_i] = f;
	}
}

__global__ void coulombErfc_kernel(int* d_pairsCount, int* d_pairsList, float alpha, float kc, float cutoff, int widthTot){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 r1 = tex1Dfetch(t_coord, d_i);
		float q1 = tex1Dfetch(t_charges, d_i);
		float4 r2;
		float4 f = c_mdd.d_force[d_i];
		int p, j;
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*widthTot + d_i];
			r2 = tex1Dfetch(t_coord, j);

			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;
			float q2 = tex1Dfetch(t_charges, j);

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float rsq = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
			float r = sqrtf(rsq);

			if(r < cutoff){

				float mult = erfcf(alpha*r)/r + alpha*(2.0f/sqrtf(M_PI))*expf(-alpha*alpha*rsq);

				mult *= -kc*q1*q2/rsq;

				f.x += mult*r2.x;
				f.y += mult*r2.y;
				f.z += mult*r2.z;
			}
		}
		c_mdd.d_force[d_i] = f;
	}
}

void Coulomb::compute(){
	if(alpha == 0.0f){
		coulomb_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, kc, cutoff, mdd->widthTot);
	} else {
		coulombErfc_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, alpha, kc, cutoff, mdd->widthTot);
	}

/*	int i;
	cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	FILE* file = fopen("coul_forces.dat", "w");
	for(i = 0; i < mdd->N; i++){
		fprintf(file, "%d %f %f %f\n", i, mdd->h_force[i].x, mdd->h_force[i].y, mdd->h_force[i].z);
	}
	fclose(file);
	exit(0);*/
}

__global__ void coulombEnergy_kernel(int* d_pairsCount, int* d_pairsList, float* d_energy, float alpha, float kc, float cutoff, int widthTot){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 r1 = tex1Dfetch(t_coord, d_i);
		float q1 = tex1Dfetch(t_charges, d_i);
		float4 r2;
		float pot = 0.0f;
		int p, j;
		for(p = 0; p < d_pairsCount[d_i]; p++){
			j = d_pairsList[p*widthTot + d_i];
			r2 = tex1Dfetch(t_coord, j);

			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;
			float q2 = tex1Dfetch(t_charges, j);

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float r = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);

			if(r < cutoff){
				pot += kc*q1*q2*erfcf(alpha*r)/r;
			}
		}
		d_energy[d_i] = pot;
	}
}

float Coulomb::get_energies(int energy_id, int timestep){
	if(timestep != lastStepEnergyComputed){
		coulombEnergy_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, d_energy, alpha, kc, cutoff, mdd->widthTot);
		cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
		energyValue = 0.0f;
		int i;
		for(i = 0; i < mdd->N; i++){
			energyValue += h_energy[i];
		}
		energyValue /= 2.0f;
		lastStepEnergyComputed = timestep;
	}
	return energyValue;
}
