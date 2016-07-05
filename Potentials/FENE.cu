#include "FENE.cuh"

//func = 1

FENE::FENE(MDData *mdd, TOPData *topdata)
{
	this->mdd = mdd;

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	using namespace std;

	int N = mdd->N;		//atomCount
	int i, j;

//MAKE PAIRLIST

	h_pairCount = (int*)calloc(N, sizeof(int));
	h_pairMap_atom = (int*)calloc((N*10), sizeof(int));	//TODO
	h_pairMap_dist = (float*)calloc((N*10), sizeof(float));

	cudaMalloc((void**)&d_pairCount, N*sizeof(int));
	cudaMalloc((void**)&d_pairMap_atom, (N*10)*sizeof(int));
	cudaMalloc((void**)&d_pairMap_dist, (N*10)*sizeof(float));

	int b;
	for (b = 0; b < topdata->bondCount; b++){
		i = topdata->bonds[b].i;
		j = topdata->bonds[b].j;
		if (topdata->bonds[b].func == 1){
			h_pairMap_atom[i + h_pairCount[i]*N] = j;
			h_pairMap_dist[i + h_pairCount[i]*N] = topdata->bonds[b].c0;
			h_pairCount[i]++;

			h_pairMap_atom[j + h_pairCount[j]*N] = i;
			h_pairMap_dist[j + h_pairCount[j]*N] = topdata->bonds[b].c0;
			h_pairCount[j]++;

			//printf("Adding %d-%d\n", i, j);
		}
	}

	cudaMemcpy(d_pairCount, h_pairCount, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_atom, h_pairMap_atom, (N*10)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pairMap_dist, h_pairMap_dist, (N*10)*sizeof(float), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(N, sizeof(float));
	cudaMalloc((void**)&d_energy, N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, N*sizeof(float), cudaMemcpyHostToDevice);

}

FENE::~FENE(){
	free(h_pairCount);
	free(h_pairMap_atom);
	free(h_pairMap_dist);
	free(h_energy);
	cudaFree(d_pairCount);
	cudaFree(d_pairMap_atom);
	cudaFree(d_pairMap_dist);
	cudaFree(d_energy);
}

__global__ void FENE_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_dist){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		float r0, temp1, temp2;

		float k = 14.0;		// spring constant {N/m}
		float R = 2.0;		//tolerance to the change of the covalent bond length

		int j;
		float dist, df;
		float3 rij;
		float4 ri, rj, f;
		ri = c_mdd.d_coord[i];

		int count = d_pairCount[i];

		f = c_mdd.d_force[i];

		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			dist = d_pairMap_dist[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			r0 = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			//FENE
			temp1 = r0 - dist;
			temp2 = R*R - temp1*temp1;
			df = -k*R*(temp1/temp2)/r0;

			f.x += df*rij.x;
			f.y += df*rij.y;
			f.z += df*rij.z;
		}
		
		c_mdd.d_force[i] = f;
	}
}

void FENE::compute(MDData *mdd){
	//checkError("before FENE potential");		//TODO
	//printf("Here!\n");
	FENE_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_dist);
	//cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);

}

__global__ void FENE_Energy_kernel(int* d_pairCount, int* d_pairMap_atom, float* d_pairMap_dist, float* d_energy){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){
		float r0, temp1;

		float k = 14.0;		//spring constant {N/m}
		float R = 2.0;		//tolerance to the change of the covalent bond length

		int j;
		float dist;
		float3 rij;
		float4 ri, rj;
		ri = c_mdd.d_coord[i];

		int count = d_pairCount[i];

		float energy = 0.0;

		int p;			//p - pairs
		for (p = 0; p < count; p++){

			j = d_pairMap_atom[i + p*c_mdd.N];
			dist = d_pairMap_dist[i + p*c_mdd.N];

			rj = c_mdd.d_coord[j];

			rij.x = rj.x - ri.x;
			rij.y = rj.y - ri.y;
			rij.z = rj.z - ri.z;

			r0 = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

			//ENERGY
			temp1 = (r0 - dist)*(r0 - dist);
			//TODO
			energy += -k*R*logf(1.0 - temp1/(R*R))/2;
		}

		d_energy[i] = energy;
	}
}


float FENE::get_energies(int energy_id, int timestep){

	FENE_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_pairCount, d_pairMap_atom, d_pairMap_dist, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;
	int i;
	for (i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}









