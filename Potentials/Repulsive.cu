#include "Repulsive.cuh"

// ![pairs] && ![bonds]

Repulsive::Repulsive(MDData *mdd, PairlistUpdater *plist, float nbCutoff, float rep_eps, float rep_sigm){

	this->mdd = mdd;
	this->plist = plist;
	this->nbCutoff = nbCutoff;
	this->rep_eps = rep_eps;
	this->rep_sigm = rep_sigm;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

Repulsive::~Repulsive(){
	free(h_energy);
	cudaFree(d_energy);
}

__global__ void Repulsive_kernel(int* pairs_count, int* pairs_list, float nbCutoff, float eps, float sigm){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;

		float temp;
		float rij_mod, df;
		float3 rij;
		float4 ri, rj, f;

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		int count = pairs_count[i];
		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = pairs_list[i + p*c_mdd.widthTot];		//TODO widthTot ??

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			if (rij_mod <= nbCutoff){

				//REPULSIVE POTENTIAL
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

	Repulsive_kernel<<<this->blockCount, this->blockSize>>>(plist->d_pairs.count, plist->d_pairs.list, nbCutoff, rep_eps, rep_sigm);
}

__global__ void Repulsive_Energy_kernel(float* d_energy, int* pairs_count, int* pairs_list, float nbCutoff, float eps, float sigm){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;

		float rij_mod, energy;
		float3 rij;
		float4 ri, rj;

		ri = c_mdd.d_coord[i];

		int count = pairs_count[i];
		energy = 0.0;
		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = pairs_list[i + p*c_mdd.widthTot];		//TODO widthTot ??

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			if (rij_mod <= nbCutoff){

				//ENERGY
				energy += eps*pow(sigm, 6.0f)/pow(rij_mod, 6.0f);
			}
		}

		d_energy[i] = energy;
	}
}

float Repulsive::get_energies(int energy_id, int timestep){

	Repulsive_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_energy, plist->d_pairs.count, plist->d_pairs.list, nbCutoff, rep_eps, rep_sigm);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
