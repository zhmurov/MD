#include "BondHarmonic.cuh"

BondHarmonic::BondHarmonic(MDData *mdd, int count, int2* bonds, float* bondsR0, float* bondsKs){
	this->mdd = mdd;

	blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int i, j, b;

//maxBonds
	h_bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bondCount, mdd->N*sizeof(int));
	for(b = 0; b < count; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		h_bondCount[i]++;
		h_bondCount[j]++;
	}
	maxBonds = 0;
	for(i = 0; i < mdd->N; i++){
		if(h_bondCount[i] > maxBonds){
			maxBonds = h_bondCount[i];
		}
		h_bondCount[i] = 0;
	}
	printf("Max bonds per atom (BondHarmonic potential) = %2d\n", maxBonds);

//bondMap
	h_bondMap = (int*)calloc((mdd->N*maxBonds), sizeof(int));
	cudaMalloc((void**)&d_bondMap, (mdd->N*maxBonds)*sizeof(int));
	h_bondMapR0 = (float*)calloc((mdd->N*maxBonds), sizeof(float));
	cudaMalloc((void**)&d_bondMapR0, (mdd->N*maxBonds)*sizeof(float));
	h_bondMapKs = (float*)calloc((mdd->N*maxBonds), sizeof(float));
	cudaMalloc((void**)&d_bondMapKs, (mdd->N*maxBonds)*sizeof(float));

	for (b = 0; b < count; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		h_bondMap[i + h_bondCount[i]*mdd->N] = j;
		h_bondMapR0[i + h_bondCount[i]*mdd->N] = bondsR0[b];
		h_bondMapKs[i + h_bondCount[i]*mdd->N] = bondsKs[b];
		h_bondCount[i]++;

		h_bondMap[j + h_bondCount[j]*mdd->N] = i;
		h_bondMapR0[j + h_bondCount[j]*mdd->N] = bondsR0[b];
		h_bondMapKs[j + h_bondCount[j]*mdd->N] = bondsKs[b];
		h_bondCount[j]++;
	}
	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMap, h_bondMap, (mdd->N*maxBonds)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMapR0, h_bondMapR0, (mdd->N*maxBonds)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMapKs, h_bondMapKs, (mdd->N*maxBonds)*sizeof(float), cudaMemcpyHostToDevice);

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

BondHarmonic::~BondHarmonic(){
	free(h_bondCount);
	free(h_bondMap);
	free(h_bondMapR0);
	free(h_bondMapKs);
	free(h_energy);
	cudaFree(d_bondCount);
	cudaFree(d_bondMap);
	cudaFree(d_bondMapR0);
	cudaFree(d_bondMapKs);
	cudaFree(d_energy);
}

//================================================================================================
__global__ void bondHarmonic_kernel(int* d_bondCount, int* d_bondMap, float* d_bondMapR0, float* d_bondMapKs){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float rij_mod, r0, ks, df;
		float3 rij;
		float4 ri, rj, f;

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		for(int b = 0; b < d_bondCount[i]; b++){
			j = d_bondMap[i + b*c_mdd.N];
			r0 = d_bondMapR0[i + b*c_mdd.N];
			ks = d_bondMapKs[i + b*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			df = ks*(rij_mod - r0)/rij_mod;

			f.x += df*rij.x;
			f.y += df*rij.y;
			f.z += df*rij.z;
		}
		c_mdd.d_force[i] = f;
	}
}

//================================================================================================
__global__ void bondHarmonicEnergy_kernel(int* d_bondCount, int* d_bondMap, float* d_bondMapR0, float* d_bondMapKs, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float rij_mod, r0, ks;
		float3 rij;
		float4 ri, rj;
		float energy = 0.0f;

		ri = c_mdd.d_coord[i];

		for(int b = 0; b < d_bondCount[i]; b++){
			j = d_bondMap[i + b*c_mdd.N];
			r0 = d_bondMapR0[i + b*c_mdd.N];
			ks = d_bondMapKs[i + b*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			energy += ks*(rij_mod - r0)*(rij_mod - r0)/2.0f;
		}
		d_energy[i] = energy;
	}
}

//================================================================================================
void BondHarmonic::compute(){
	bondHarmonic_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bondMap, d_bondMapR0, d_bondMapKs);
}

//================================================================================================
float BondHarmonic::getEnergies(int energyId, int timestep){
	bondHarmonicEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bondMap, d_bondMapR0, d_bondMapKs, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0f;
	for(int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
