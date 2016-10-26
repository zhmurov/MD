#include "FENE.cuh"

//[bonds] func = 1

FENE::FENE(MDData *mdd, int bondCount, int2* bonds, float* bondsC0)
{
	this->mdd = mdd;

	blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int i, j, b;

//max_pairCount

	int* temp_pairCount;
	temp_pairCount = (int*)calloc(mdd->N, sizeof(int));

	for (b = 0; b < bondCount; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		//if (topdata->bonds[b].func == 1){ //TODO
			temp_pairCount[i]++;
			temp_pairCount[j]++;
		//}
	}

	max_pairCount = 0;

	for (i = 0; i < mdd->N; i++){

		if (temp_pairCount[i] > max_pairCount){
			max_pairCount = temp_pairCount[i];
		}
	}
	printf("max_pairCount = %2d\n", max_pairCount);

	free(temp_pairCount);

//MAKE PAIRLIST

	h_pairCount = (int*)calloc(mdd->N, sizeof(int));
	h_pairMap_atom = (int*)calloc((mdd->N*max_pairCount), sizeof(int));	//TODO
	h_pairMap_r0 = (float*)calloc((mdd->N*max_pairCount), sizeof(float));

	cudaMalloc((void**)&d_pairCount, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_pairMap_atom, (mdd->N*max_pairCount)*sizeof(int));
	cudaMalloc((void**)&d_pairMap_r0, (mdd->N*max_pairCount)*sizeof(float));

	for (b = 0; b < bondCount; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		//if (topdata->bonds[b].func == 1){
			h_pairMap_atom[i + h_pairCount[i]*mdd->N] = j;
			h_pairMap_r0[i + h_pairCount[i]*mdd->N] = bondsC0[b];
			h_pairCount[i]++;

			h_pairMap_atom[j + h_pairCount[j]*mdd->N] = i;
			h_pairMap_r0[j + h_pairCount[j]*mdd->N] = bondsC0[b];
			h_pairCount[j]++;

			//printf("Adding %2d-%2d\n", i, j);
		//}
	}

	cudaMemcpy(d_pairCount, h_pairCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_atom, h_pairMap_atom, (mdd->N*max_pairCount)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_r0, h_pairMap_r0, (mdd->N*max_pairCount)*sizeof(float), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

FENE::~FENE(){
	free(h_pairCount);
	free(h_pairMap_atom);
	free(h_pairMap_r0);
	free(h_energy);
	cudaFree(d_pairCount);
	cudaFree(d_pairMap_atom);
	cudaFree(d_pairMap_r0);
	cudaFree(d_energy);
}

//================================================================================================
__global__ void FENE_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		float k = 8370;		//spring constant			 [kJ / (mol*nm^2)]
		float R = 0.2;		//tolerance to the change of the covalent bond length [nm]

		int j;

//RENAMED r0 -> rij_mod
//RENAMED dist -> r0

		float temp1, temp2;
		float rij_mod, r0, df;
		float3 rij;
		float4 ri, rj, f;

		ri = c_mdd.d_coord[i];
		f = c_mdd.d_force[i];

		int p;			//p - pairs
		for (p = 0; p < d_pairCount[i]; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			r0 = d_pairMap_r0[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			float3 pb = c_mdd.bc.len;
			rij.x -= rint(rij.x/pb.x)*pb.x;
			rij.y -= rint(rij.y/pb.y)*pb.y;
			rij.z -= rint(rij.z/pb.z)*pb.z;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			//FENE POTENTIAL
			temp1 = rij_mod - r0;
			temp2 = R*R - temp1*temp1;

			df = k*R*R*(temp1/(temp2*rij_mod));

			f.x += df*rij.x;
			f.y += df*rij.y;
			f.z += df*rij.z;
		}
		
		c_mdd.d_force[i] = f;

	}
}

//================================================================================================
__global__ void FENE_Energy_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_r0, float* d_energy){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		float k = 8370;		//spring constant			 [kJ / (mol*nm^2)]
		float R = 0.2;		//tolerance to the change of the covalent bond length [nm]

		int j;

		float temp1;
		float rij_mod, r0;
		float3 rij;
		float4 ri, rj;

		ri = c_mdd.d_coord[i];

		float energy = 0.0;

		int p;			//p - pairs
		for (p = 0; p < d_pairCount[i]; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			r0 = d_pairMap_r0[i + p*c_mdd.N];

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
			temp1 = (rij_mod - r0)*(rij_mod - r0);

			energy += -k*R*R*logf(1.0 - temp1/(R*R))/2;
		}

		d_energy[i] = energy;
	}
}

//================================================================================================
void FENE::compute(){

	FENE_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0);

}

//================================================================================================
float FENE::get_energies(int energy_id, int timestep){

	FENE_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_r0, d_energy);


	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){

		energy_sum += h_energy[i];
	}

	return energy_sum;
}
