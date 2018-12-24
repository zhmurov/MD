#include "HarmonicFixation.cuh"

HarmonicFixation::HarmonicFixation(MDData* mdd, float3* fixedAtomsR0, float* ks){

	this->mdd = mdd;

	h_fixedAtomsR0 = (float3*)calloc(mdd->N, sizeof(float3));
	h_ks = (float*)calloc(mdd->N, sizeof(float));

	memcpy(h_fixedAtomsR0, fixedAtomsR0, mdd->N*sizeof(float3));
	memcpy(h_ks, ks, mdd->N*sizeof(float));
	
	cudaMalloc((void**)&d_fixedAtomsR0, mdd->N*sizeof(float3));
	cudaMalloc((void**)&d_ks, mdd->N*sizeof(float));

	cudaMemcpy(d_fixedAtomsR0, h_fixedAtomsR0, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ks, h_ks, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

	averFmod = 0.0f;		// average Fmod

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;


// force
	h_fmod = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_fmod, mdd->N*sizeof(float));
	cudaMemcpy(d_fmod, h_fmod, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

HarmonicFixation::~HarmonicFixation(){
	free(h_fixedAtomsR0);
	free(h_ks);
	free(h_energy);
	cudaFree(d_energy);
	cudaFree(d_fixedAtomsR0);
	cudaFree(d_ks);
}

__global__ void harmonic_fixation_kernel(float3* d_fixedAtomsR0, float* d_ks, float* d_fmod){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){

		float3 r0 = d_fixedAtomsR0[i];
		float4 ri = c_mdd.d_coord[i];
		float4 f = c_mdd.d_force[i];

		float ks = d_ks[i];

		float4 dr;

		dr.x = ri.x - r0.x;
		dr.y = ri.y - r0.y;
		dr.z = ri.z - r0.z;
		
		f.x += ks*dr.x;
		f.y += ks*dr.y;
		f.z += ks*dr.z;

		c_mdd.d_force[i] = f;

		dr.w = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
		d_fmod[i] = ks*dr.w;
	}
}

void HarmonicFixation::compute(){

	harmonic_fixation_kernel<<<this->blockCount, this->blockSize>>>(d_fixedAtomsR0, d_ks, d_fmod);

	cudaMemcpy(h_fmod, d_fmod, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < mdd->N; i++){
		averFmod += h_fmod[i];
	}
}

__global__ void HarmonicFixationEnergykernel(float3* d_fixedAtomsR0, float* d_ks, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){

		float3 r0 = d_fixedAtomsR0[i];	// initial coordinates of base
		float4 ri = c_mdd.d_coord[i];

		float ks = d_ks[i];

		float4 rij;
		// ri - current coordinates of 'i' atom
		// r0 - fixef coordinate of 'i' atom
		// rij - distance between ri and r0
		rij.x = ri.x - r0.x;
		rij.y = ri.y - r0.y;
		rij.z = ri.z - r0.z;

		rij.w = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

		d_energy[i] = ks*(rij.w-0.0f)*(rij.w-0.0f)/2.0f;
	}
}

float HarmonicFixation::getEnergies(int energyId, int timestep){
	HarmonicFixationEnergykernel<<<this->blockCount, this->blockSize>>>(d_fixedAtomsR0, d_ks, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0f;

	for(int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
