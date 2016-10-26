#include "LJP.cuh"

//[pairs] func = 1

LJP::LJP(MDData *mdd, int pairsCount, int2* pairs, float* pairsC0, float* pairsC1){

	this->mdd = mdd;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;

	int i, j, b;

//MAKE PAIRLIST

	h_pairCount = (int*)calloc(mdd->N, sizeof(int));
	h_pairMap_atom = (int*)calloc((mdd->N*mdd->N), sizeof(int));
	h_pairMap_r0 = (float*)calloc((mdd->N*mdd->N), sizeof(float));
	h_pairMap_eps = (float*)calloc((mdd->N*mdd->N), sizeof(float));

	cudaMalloc((void**)&d_pairCount, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_pairMap_atom, (mdd->N*mdd->N)*sizeof(int));
	cudaMalloc((void**)&d_pairMap_r0, (mdd->N*mdd->N)*sizeof(float));
	cudaMalloc((void**)&d_pairMap_eps, (mdd->N*mdd->N)*sizeof(float));

	for (b = 0; b < pairsCount; b++){
		i = pairs[b].x;
		j = pairs[b].y;

		//if (topdata->pairs[b].func == 1){
			h_pairMap_atom[i + h_pairCount[i]*mdd->N] = j;
			h_pairMap_r0[i + h_pairCount[i]*mdd->N] = pairsC0[b];
			h_pairMap_eps[i + h_pairCount[i]*mdd->N] = pairsC1[b];
			h_pairCount[i]++;

			h_pairMap_atom[j + h_pairCount[j]*mdd->N] = i;
			h_pairMap_r0[j + h_pairCount[j]*mdd->N] = pairsC0[b];
			h_pairMap_eps[j + h_pairCount[j]*mdd->N] = pairsC1[b];
			h_pairCount[j]++;

			//printf("Adding %2d-%2d;\teps = %7.7f\n", i, j, h_pairMap_eps[i + h_pairCount[j]*mdd->N]);
		//}
	}

/*
	for (i = 0; i < mdd->N; i++){

		int N = h_pairCount[i];
		for (j = 0; j < N; j++){

			printf("Adding %2d-%2d;\teps = %7.7f\n", i, h_pairMap_atom[i + j*mdd->N], h_pairMap_eps[i + j*mdd->N]);
		}
		printf("\n");
	}
*/

	cudaMemcpy(d_pairCount, h_pairCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_atom, h_pairMap_atom, (mdd->N*mdd->N)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_r0, h_pairMap_r0, (mdd->N*mdd->N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_eps, h_pairMap_eps, (mdd->N*mdd->N)*sizeof(float), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

LJP::~LJP(){
	free(h_pairCount);
	free(h_pairMap_atom);
	free(h_pairMap_r0);
	free(h_pairMap_eps);
	free(h_energy);
	cudaFree(d_pairCount);
	cudaFree(d_pairMap_atom);
	cudaFree(d_pairMap_r0);
	cudaFree(d_pairMap_eps);
	cudaFree(d_energy);
}

__global__ void LJP_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0, float* d_pairMap_eps){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){
		
		int j;

		float temp1, temp2;
		float rij_mod, df, r0, eps;
		float3 rij;
		float4 ri, rj, f;

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		int count = d_pairCount[i];
		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			r0 = d_pairMap_r0[i + p*c_mdd.N];
			eps = d_pairMap_eps[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			//LENNARD-JONES POTENTIAL
			temp1 = pow(r0, 12.0f)*(-12.0f)/pow(rij_mod, 13.0f);
			temp2 = pow(r0, 6.0f)*(-6.0f)/pow(rij_mod, 7.0f);

			df = eps*(temp1 - 2.0*temp2)/rij_mod;

			f.x += df*rij.x;
			f.y += df*rij.y;
			f.z += df*rij.z;
		}

		c_mdd.d_force[i] = f;
	}
}



void LJP::compute(){

	LJP_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0, d_pairMap_eps);

}

__global__ void LJP_Energy_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0, float* d_pairMap_eps, float* d_energy){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;

		float temp1, temp2;
		float rij_mod, r0, eps, energy;
		float3 rij;
		float4 ri, rj;	

		ri = c_mdd.d_coord[i];

		int count = d_pairCount[i];
		energy = 0.0f;
		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			r0 = d_pairMap_r0[i + p*c_mdd.N];
			eps = d_pairMap_eps[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			//ENERGY
			temp1 = pow(r0, 12.0f)/pow(rij_mod, 12.0f);
			temp2 = pow(r0, 6.0f)/pow(rij_mod, 6.0f);

			energy += -eps*(temp1 - 2.0*temp2);
		}

		d_energy[i] = energy;
	}
}

float LJP::get_energies(int energy_id, int timestep){

	LJP_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0, d_pairMap_eps, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}

	return energy_sum;
}
