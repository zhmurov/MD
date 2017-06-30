#include "FENE.cuh"

//[bonds] func = 1

FENE::FENE(MDData *mdd, float Ks, float R, int Count, int2* Bonds, float* BondsR0){
	this->mdd = mdd;
	this->Ks = Ks;
	this->R = R;

	blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int i, j, b;

// maxN
	h_bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bondCount, mdd->N*sizeof(int));
	for(b = 0; b < Count; b++){
		i = Bonds[b].x;
		j = Bonds[b].y;

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
	printf("Max bonds per atom = %2d\n", maxBonds);

// bondMap
	h_bondMap = (int*)calloc((mdd->N*maxBonds), sizeof(int));
	cudaMalloc((void**)&d_bondMap, (mdd->N*maxBonds)*sizeof(int));
	h_bondMapR0 = (float*)calloc((mdd->N*maxBonds), sizeof(float));
	cudaMalloc((void**)&d_bondMapR0, (mdd->N*maxBonds)*sizeof(float));

	for (b = 0; b < Count; b++){
		i = Bonds[b].x;
		j = Bonds[b].y;

		h_bondMap[i + h_bondCount[i]*mdd->N] = j;
		h_bondMapR0[i + h_bondCount[i]*mdd->N] = BondsR0[b];
		h_bondCount[i]++;

		h_bondMap[j + h_bondCount[j]*mdd->N] = i;
		h_bondMapR0[j + h_bondCount[j]*mdd->N] = BondsR0[b];
		h_bondCount[j]++;
	}
	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMap, h_bondMap, (mdd->N*maxBonds)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMapR0, h_bondMapR0, (mdd->N*maxBonds)*sizeof(float), cudaMemcpyHostToDevice);

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

FENE::~FENE(){
	free(h_bondCount);
	free(h_bondMap);
	free(h_bondMapR0);
	free(h_energy);
	cudaFree(d_bondCount);
	cudaFree(d_bondMap);
	cudaFree(d_bondMapR0);
	cudaFree(d_energy);
}

//================================================================================================
__global__ void fene_kernel(float ks, float R, int* d_bondCount, int* d_bondMap, float* d_bondMapR0){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float temp1, temp2;
		float rij_mod, r0, df;
		float3 rij;
		float4 ri, rj, f;

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		for(int b = 0; b < d_bondCount[i]; b++){
			j = d_bondMap[i + b*c_mdd.N];
			r0 = d_bondMapR0[i + b*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			temp1 = rij_mod - r0;
			temp2 = R*R - temp1*temp1;

			df = ks*R*R*(temp1/(temp2*rij_mod));

			f.x += df*rij.x;
			f.y += df*rij.y;
			f.z += df*rij.z;
		}

		c_mdd.d_force[i] = f;
	}
}

//================================================================================================
__global__ void feneEnergy_kernel(float ks, float R, int* d_bondCount, int* d_bondMap, float* d_bondMapR0, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float temp1;
		float rij_mod, r0;
		float3 rij;
		float4 ri, rj;
		float energy = 0.0f;

		ri = c_mdd.d_coord[i];

		for(int b = 0; b < d_bondCount[i]; b++){

			j = d_bondMap[i + b*c_mdd.N];
			r0 = d_bondMapR0[i + b*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			temp1 = (rij_mod - r0)*(rij_mod - r0);
			energy += -ks*R*R*logf(1.0 - temp1/(R*R))/2;
		}

		d_energy[i] = energy;
	}
}

//================================================================================================
void FENE::compute(){
	fene_kernel<<<this->blockCount, this->blockSize>>>(Ks, R, d_bondCount, d_bondMap, d_bondMapR0);
}

//================================================================================================
float FENE::getEnergies(int energyId, int timestep){
	feneEnergy_kernel<<<this->blockCount, this->blockSize>>>(Ks, R, d_bondCount, d_bondMap, d_bondMapR0, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
