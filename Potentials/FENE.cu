#include "FENE.cuh"

//[bonds] func = 1

FENE::FENE(MDData *mdd, int bondCount, int2* bonds, float* bonds_C0)
{
	this->mdd = mdd;

	blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int i, j, b;

//MAX_NBONDS
	int* Nbonds;		//quantity of bonds for an aminoacid
	Nbonds = (int*)calloc(mdd->N, sizeof(int));
	for (b = 0; b < bondCount; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		Nbonds[i]++;
		Nbonds[j]++;
	}
	max_Nbonds = 0;
	for (i = 0; i < mdd->N; i++){
		if (Nbonds[i] > max_Nbonds){
			max_Nbonds = Nbonds[i];
		}
	}
	printf("max_Nbonds = %2d\n", max_Nbonds);
	free(Nbonds);

//MAKE PAIRLIST

	h_bondCount = (int*)calloc(mdd->N, sizeof(int));
	h_bondMap_atom = (int*)calloc((mdd->N*max_Nbonds), sizeof(int));	//TODO
	h_bondMap_r0 = (float*)calloc((mdd->N*max_Nbonds), sizeof(float));

	cudaMalloc((void**)&d_bondCount, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_bondMap_atom, (mdd->N*max_Nbonds)*sizeof(int));
	cudaMalloc((void**)&d_bondMap_r0, (mdd->N*max_Nbonds)*sizeof(float));

	for (b = 0; b < bondCount; b++){
		i = bonds[b].x;
		j = bonds[b].y;

		h_bondMap_atom[i + h_bondCount[i]*mdd->N] = j;
		h_bondMap_r0[i + h_bondCount[i]*mdd->N] = bonds_C0[b];
		h_bondCount[i]++;

		h_bondMap_atom[j + h_bondCount[j]*mdd->N] = i;
		h_bondMap_r0[j + h_bondCount[j]*mdd->N] = bonds_C0[b];
		h_bondCount[j]++;
	}

	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMap_atom, h_bondMap_atom, (mdd->N*max_Nbonds)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondMap_r0, h_bondMap_r0, (mdd->N*max_Nbonds)*sizeof(float), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

FENE::~FENE(){
	free(h_bondCount);
	free(h_bondMap_atom);
	free(h_bondMap_r0);
	free(h_energy);
	cudaFree(d_bondCount);
	cudaFree(d_bondMap_atom);
	cudaFree(d_bondMap_r0);
	cudaFree(d_energy);
}

//================================================================================================
__global__ void FENE_kernel(int* d_bondCount, int* d_bondMap_atom, float* d_bondMap_r0){

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

		int b;			//b - bonds
		for (b = 0; b < d_bondCount[i]; b++){

			j = d_bondMap_atom[i + b*c_mdd.N];
			r0 = d_bondMap_r0[i + b*c_mdd.N];

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
__global__ void FENE_Energy_kernel(int* d_bondCount, int* d_bondMap_atom, float* d_bondMap_r0, float* d_energy){

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

		int p;			//b - bonds
		for (p = 0; p < d_bondCount[i]; p++){

			j = d_bondMap_atom[i + p*c_mdd.N];
			r0 = d_bondMap_r0[i + p*c_mdd.N];

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

	FENE_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bondMap_atom, d_bondMap_r0);

}

//================================================================================================
float FENE::get_energies(int energy_id, int timestep){

	FENE_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bondMap_atom, d_bondMap_r0, d_energy);


	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){

		energy_sum += h_energy[i];
	}

	return energy_sum;
}
