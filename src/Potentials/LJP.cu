#include "LJP.cuh"

// top.pairs

LJP::LJP(MDData *mdd, int count, int2* pairs, float* pairsR0, float* pairsEps){
	this->mdd = mdd;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;

	int i, j, p;

// maxPairs
	h_pairCount = (int*)calloc(mdd->N, sizeof(int));
	for(p = 0; p < count; p++){
		i = pairs[p].x;
		j = pairs[p].y;

		h_pairCount[i]++;
		h_pairCount[j]++;
	}
	maxPairs = 0;
	for(i = 0; i < mdd->N; i++){
		if(h_pairCount[i] > maxPairs){
			maxPairs = h_pairCount[i];
		}
		h_pairCount[i] = 0;
	}
	printf("Max pairs per atom = %2d\n", maxPairs);

// pairMap
	h_pairCount = (int*)calloc(mdd->N, sizeof(int));
	h_pairMap = (int*)calloc((mdd->N*maxPairs), sizeof(int));
	h_pairMapR0 = (float*)calloc((mdd->N*maxPairs), sizeof(float));
	h_pairMapEps = (float*)calloc((mdd->N*maxPairs), sizeof(float));

	cudaMalloc((void**)&d_pairCount, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_pairMap, (mdd->N*maxPairs)*sizeof(int));
	cudaMalloc((void**)&d_pairMapR0, (mdd->N*maxPairs)*sizeof(float));
	cudaMalloc((void**)&d_pairMapEps, (mdd->N*maxPairs)*sizeof(float));

	for(p = 0; p < count; p++){
		i = pairs[p].x;
		j = pairs[p].y;

		h_pairMap[i + h_pairCount[i]*mdd->N] = j;
		h_pairMapR0[i + h_pairCount[i]*mdd->N] = pairsR0[p];
		h_pairMapEps[i + h_pairCount[i]*mdd->N] = pairsEps[p];
		h_pairCount[i]++;

		h_pairMap[j + h_pairCount[j]*mdd->N] = i;
		h_pairMapR0[j + h_pairCount[j]*mdd->N] = pairsR0[p];
		h_pairMapEps[j + h_pairCount[j]*mdd->N] = pairsEps[p];
		h_pairCount[j]++;
	}

	cudaMemcpy(d_pairCount, h_pairCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap, h_pairMap, (mdd->N*maxPairs)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMapR0, h_pairMapR0, (mdd->N*maxPairs)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMapEps, h_pairMapEps, (mdd->N*maxPairs)*sizeof(float), cudaMemcpyHostToDevice);

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

LJP::~LJP(){
	free(h_pairCount);
	free(h_pairMap);
	free(h_pairMapR0);
	free(h_pairMapEps);
	free(h_energy);
	cudaFree(d_pairCount);
	cudaFree(d_pairMap);
	cudaFree(d_pairMapR0);
	cudaFree(d_pairMapEps);
	cudaFree(d_energy);
}

__global__ void lj_kernel(int* d_pairCount, int* d_pairMap, float* d_pairMapR0, float* d_pairMapEps){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float temp1, temp2;
		float rij_mod, df, r0, eps;
		float3 rij;
		float4 ri, rj, f;
		int count = d_pairCount[i];

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		for(int p = 0; p < count; p++){
			j = d_pairMap[i + p*c_mdd.N];
			r0 = d_pairMapR0[i + p*c_mdd.N];
			eps = d_pairMapEps[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

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
	lj_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap, d_pairMapR0, d_pairMapEps);
}

__global__ void ljEnergy_kernel(int* d_pairCount, int* d_pairMap, float* d_pairMapR0, float* d_pairMapEps, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		int j;
		float temp1, temp2;
		float rij_mod, r0, eps;
		float3 rij;
		float4 ri, rj;
		int count = d_pairCount[i];
		float energy = 0.0f;

		ri = c_mdd.d_coord[i];

		for(int p = 0; p < count; p++){
			j = d_pairMap[i + p*c_mdd.N];
			r0 = d_pairMapR0[i + p*c_mdd.N];
			eps = d_pairMapEps[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			temp1 = pow(r0, 12.0f)/pow(rij_mod, 12.0f);
			temp2 = pow(r0, 6.0f)/pow(rij_mod, 6.0f);

			energy += eps*(temp1 - 2.0*temp2);
		}
		d_energy[i] = energy;
	}
}

float LJP::getEnergies(int energyId, int timestep){
	ljEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap, d_pairMapR0, d_pairMapEps, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
