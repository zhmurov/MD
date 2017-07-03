#include "Pulling.cuh"

Pulling::Pulling(MDData* mdd, float3* h_baseR0, int baseFreq, float vel, float3* h_n, float* h_ks, int dcdFreq){
	this->mdd = mdd;
	this->h_baseR0 = h_baseR0;
	this->baseFreq = baseFreq;
	this->vel = vel;
	this->h_n = h_n;
	this->h_ks = h_ks;
	this->dcdFreq = dcdFreq;

	cudaMalloc((void**)&d_baseR0, mdd->N*sizeof(float3));
	cudaMalloc((void**)&d_n, mdd->N*sizeof(float3));
	cudaMalloc((void**)&d_ks, mdd->N*sizeof(float));
	cudaMemcpy(d_baseR0, h_baseR0, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ks, h_ks, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

	//TODO
	//this->baseDisplacement = baseDisplacement;
	baseDisplacement = 0.0f;

	// average Fmod
	//this->averFmod = averFmod;
	averFmod = 0.0f;

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	//TODO
	FILE* data = fopen("force_extension.out", "w");
	fclose(data);

// force
	h_fmod = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_fmod, mdd->N*sizeof(float));
	cudaMemcpy(d_fmod, h_fmod, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

// energy
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

Pulling::~Pulling(){
	free(h_energy);
	cudaFree(d_energy);
	cudaFree(d_baseR0);
	cudaFree(d_n);
	cudaFree(d_ks);
}

__global__ void pulling_kernel(float3* d_baseR0, float baseDisplacement, float3* d_n, float* d_ks, float* d_fmod){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){

		float3 r0 = d_baseR0[i];
		float4 ri = c_mdd.d_coord[i];
		float4 f = c_mdd.d_force[i];

		float ks = d_ks[i];
		float3 n = d_n[i];

		float4 dr;

		dr.x = r0.x + n.x*baseDisplacement - ri.x;
		dr.y = r0.y + n.y*baseDisplacement - ri.y;
		dr.z = r0.z + n.z*baseDisplacement - ri.z;

		f.x += ks*dr.x;
		f.y += ks*dr.y;
		f.z += ks*dr.z;

		c_mdd.d_force[i] = f;

		dr.w = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
		d_fmod[i] = ks*dr.w;
	}
}

void Pulling::compute(){
	if(mdd->step % baseFreq == 0){
		baseDisplacement = vel*mdd->dt*mdd->step;
	}

	pulling_kernel<<<this->blockCount, this->blockSize>>>(d_baseR0, baseDisplacement, d_n, d_ks, d_fmod);

	cudaMemcpy(h_fmod, d_fmod, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < mdd->N; i++){
		averFmod += h_fmod[i];
	}

// output
	if(mdd->step % dcdFreq == 0){
		float dx, dy, dz, dr;
		cudaMemcpy(mdd->h_coord, mdd->d_coord, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);

		for(int i = 0; i < mdd->N; i++){
			if(h_ks[i] > 0.0f){
				dx = mdd->h_coord[i].x - h_baseR0[i].x;
				dy = mdd->h_coord[i].y - h_baseR0[i].y;
				dz = mdd->h_coord[i].z - h_baseR0[i].z;
				dr = sqrtf(dx*dx + dy*dy + dz*dz);
			}
		}
		averFmod /= float(dcdFreq);

		FILE* data = fopen("force_extension.out", "a");
		fprintf(data, "%12d\t", mdd->step);

		for(int i = 0; i < mdd->N; i++){
			if(h_ks[i] > 0.0f){
				fprintf(data, "%4.6f\t", baseDisplacement);
				fprintf(data, "%4.6f\t", averFmod);
				fprintf(data, "%4.6f", dr);
			}
		}
		fprintf(data, "\n");
		fclose(data);

		averFmod = 0.0f;
	}
}

__global__ void PullingEnergykernel(float3* d_baseR0, float baseDisplacement, float3* d_n, float* d_ks, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){

		float3 r0 = d_baseR0[i];	// initial coordinates of base
		float4 ri = c_mdd.d_coord[i];

		float ks = d_ks[i];
		float3 n = d_n[i];

		float4 rij;
		// ri - current coordinates of 'i' atom
		// rj - current coordinates of base
		// rij - distance between 'i' atom and base
		rij.x = r0.x + n.x*baseDisplacement - ri.x;
		rij.y = r0.y + n.y*baseDisplacement - ri.y;
		rij.z = r0.z + n.z*baseDisplacement - ri.z;

		rij.w = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

		d_energy[i] = ks*(rij.w-0.0f)*(rij.w-0.0f)/2.0f;
	}
}

float Pulling::getEnergies(int energyId, int timestep){
	PullingEnergykernel<<<this->blockCount, this->blockSize>>>(d_baseR0, baseDisplacement, d_n, d_ks, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0f;

	for(int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
