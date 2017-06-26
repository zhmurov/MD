#include "Pulling.cuh"

Pulling::Pulling(MDData* mdd, float3* h_base_r0, int base_freq, float vel, float3* h_n, float* h_ks, int dcd_freq){

	this->mdd = mdd;
	this->h_base_r0 = h_base_r0;
	this->base_freq = base_freq;
	this->vel = vel;
	this->h_n = h_n;
	this->h_ks = h_ks;
	this->dcd_freq = dcd_freq;

	cudaMalloc((void**)&d_base_r0, mdd->N*sizeof(float3));
	cudaMemcpy(d_base_r0, h_base_r0, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_n, mdd->N*sizeof(float3));
	cudaMemcpy(d_n, h_n, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_ks, mdd->N*sizeof(float));
	cudaMemcpy(d_ks, h_ks, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

	this->base_displacement = base_displacement;
	base_displacement = 0.0f;

	this->aver_fmod = aver_fmod;
	aver_fmod = 0.0f;		// average fmod

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	FILE* data = fopen("force_extension.out", "w");
	fclose(data);

//FORCE
	h_fmod = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_fmod, mdd->N*sizeof(float));
	cudaMemcpy(d_fmod, h_fmod, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

Pulling::~Pulling(){
	free(h_energy);
	cudaFree(d_energy);
}


__global__ void Pulling_kernel(float3* d_base_r0, float base_displacement, float3* d_n, float* d_ks, float* d_fmod){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){

		float4 ri = c_mdd.d_coord[i];
		float4 f = c_mdd.d_force[i];
		float3 r0 = d_base_r0[i];

		float ks = d_ks[i];
		float3 n = d_n[i];

		float4 dr;

		dr.x = r0.x + n.x*base_displacement - ri.x;
		dr.y = r0.y + n.y*base_displacement - ri.y;
		dr.z = r0.z + n.z*base_displacement - ri.z;

		f.x += ks*dr.x;
		f.y += ks*dr.y;
		f.z += ks*dr.z;

		c_mdd.d_force[i] = f;

		dr.w = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
		d_fmod[i] = ks*dr.w;
	}
}

void Pulling::compute(){

	if(mdd->step % base_freq == 0){
		base_displacement = vel*mdd->dt*mdd->step;
	}
	if(mdd->step % dcd_freq == 0){
		printf("base_displacement = %f\n", base_displacement);
	}

	Pulling_kernel<<<this->blockCount, this->blockSize>>>(d_base_r0, base_displacement, d_n, d_ks, d_fmod);

	cudaMemcpy(h_fmod, d_fmod, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < mdd->N; i++){
		aver_fmod += h_fmod[i];
	}

	// OUTPUT
	if(mdd->step % dcd_freq == 0){

		float dx, dy, dz, dr;
		cudaMemcpy(mdd->h_coord, mdd->d_coord, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);

		for(int i = 0; i < mdd->N; i++){
			if(h_ks[i] > 0.0f){
				dx = mdd->h_coord[i].x - h_base_r0[i].x;
				dy = mdd->h_coord[i].y - h_base_r0[i].y;
				dz = mdd->h_coord[i].z - h_base_r0[i].z;
				dr = sqrtf(dx*dx + dy*dy + dz*dz);
			}
		}
		aver_fmod /= float(dcd_freq);

		FILE* data = fopen("force_extension.out", "a");
		fprintf(data, "%12d\t", mdd->step);

		for(int i = 0; i < mdd->N; i++){
			if(h_ks[i] > 0.0f){
				fprintf(data, "%4.6f\t", base_displacement);
				fprintf(data, "%4.6f\t", aver_fmod);
				fprintf(data, "%4.6f", dr);
			}
		}
		fprintf(data, "\n");
		fclose(data);

		aver_fmod = 0.0f;
	}
}

/*
void PullingEnergykernel(){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i < c_mdd.N){
		;
	}
}
*/

float Pulling::getEnergies(int energyId, int timestep){

	//PullingEnergykernel<<<this->blockCount, this->blockSize>>>();

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0f;

	for(int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
