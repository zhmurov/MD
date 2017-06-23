#include "LJP.cuh"

//[pairs] func = 1

LJP::LJP(MDData *mdd, int pairCount, int2* pairs, float* pairs_C0, float* pairs_C1){

	this->mdd = mdd;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;

	int i, j, p;
//MAX_NPAIRS
	int* Npairs;			//quantity of pairs for an aminoacid
	Npairs = (int*)calloc(mdd->N, sizeof(int));
	for (p = 0; p < pairCount; p++){
		i = pairs[p].x;
		j = pairs[p].y;

		Npairs[i]++;
		Npairs[j]++;
	}
	max_Npairs = 0;
	for (i = 0; i < mdd->N; i++){
		if (Npairs[i] > max_Npairs){
			max_Npairs = Npairs[i];
		}
	}
	printf("max_Npairs = %2d\n", max_Npairs);
	free(Npairs);

//MAKE PAIRLIST

	h_pairCount = (int*)calloc(mdd->N, sizeof(int));
	h_pairMap_atom = (int*)calloc((mdd->N*max_Npairs), sizeof(int));
	h_pairMap_r0 = (float*)calloc((mdd->N*max_Npairs), sizeof(float));
	h_pairMap_eps = (float*)calloc((mdd->N*max_Npairs), sizeof(float));

	cudaMalloc((void**)&d_pairCount, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_pairMap_atom, (mdd->N*max_Npairs)*sizeof(int));
	cudaMalloc((void**)&d_pairMap_r0, (mdd->N*max_Npairs)*sizeof(float));
	cudaMalloc((void**)&d_pairMap_eps, (mdd->N*max_Npairs)*sizeof(float));

	for (p = 0; p < pairCount; p++){
		i = pairs[p].x;
		j = pairs[p].y;

		h_pairMap_atom[i + h_pairCount[i]*mdd->N] = j;
		h_pairMap_r0[i + h_pairCount[i]*mdd->N] = pairs_C0[p];
		h_pairMap_eps[i + h_pairCount[i]*mdd->N] = pairs_C1[p];
		h_pairCount[i]++;

		h_pairMap_atom[j + h_pairCount[j]*mdd->N] = i;
		h_pairMap_r0[j + h_pairCount[j]*mdd->N] = pairs_C0[p];
		h_pairMap_eps[j + h_pairCount[j]*mdd->N] = pairs_C1[p];
		h_pairCount[j]++;
	}

	cudaMemcpy(d_pairCount, h_pairCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_atom, h_pairMap_atom, (mdd->N*max_Npairs)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_r0, h_pairMap_r0, (mdd->N*max_Npairs)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_eps, h_pairMap_eps, (mdd->N*max_Npairs)*sizeof(float), cudaMemcpyHostToDevice);

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

__global__ void lj_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0, float* d_pairMap_eps){

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

			/*float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;*/

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

	
	lj_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0, d_pairMap_eps);

}

__global__ void ljEnergy_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0, float* d_pairMap_eps, float* d_energy){

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

			energy += eps*(temp1 - 2.0*temp2);
		}

		d_energy[i] = energy;
	}
}

float LJP::getEnergies(int energyId, int timestep){

	ljEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0, d_pairMap_eps, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}

	return energy_sum;
}
