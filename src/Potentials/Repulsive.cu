#include "Repulsive.cuh"

// !top.pairs && !top.bonds

Repulsive::Repulsive(MDData *mdd, PairlistUpdater *plist, float nbCutoff, float eps, float sigm){
	this->mdd = mdd;
	this->plist = plist;
	this->nbCutoff = nbCutoff;
	this->eps = eps;
	this->sigm = sigm;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

Repulsive::~Repulsive(){
	free(h_energy);
	cudaFree(d_energy);
}

__global__ void repulsive_kernel(int* d_pairCount, int* d_pairList, float nbCutoff, float eps, float sigm){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float temp;
		float rij_mod, df;
		float3 rij;
		float4 ri, rj, f;
		int count = d_pairCount[i];

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		for(int p = 0; p < count; p++){
			j = d_pairList[i + p*c_mdd.widthTot];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			if(rij_mod <= nbCutoff){
				temp = 6.0f*pow(sigm, 6.0f)/pow(rij_mod, 7.0f);
				df = -eps*temp/rij_mod;

				f.x += df*rij.x;
				f.y += df*rij.y;
				f.z += df*rij.z;
			}
		}
		c_mdd.d_force[i] = f;
	}
}

void Repulsive::compute(){
	repulsive_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, nbCutoff, eps, sigm);
}

__global__ void repulsiveEnergy_kernel(int* d_pairCount, int* d_pairList, float nbCutoff, float eps, float sigm, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float rij_mod;
		float3 rij;
		float4 ri, rj;
		int count = d_pairCount[i];
		float energy = 0.0f;

		ri = c_mdd.d_coord[i];

		for(int p = 0; p < count; p++){
			j = d_pairList[i + p*c_mdd.widthTot];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			if(rij_mod <= nbCutoff){
				energy += eps*pow(sigm, 6.0f)/pow(rij_mod, 6.0f);
			}
		}
		d_energy[i] = energy;
	}
}

float Repulsive::getEnergies(int energyId, int timestep){
	repulsiveEnergy_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, nbCutoff, eps, sigm, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
