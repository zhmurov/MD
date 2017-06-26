/*
 * PushingSphere.cu
 *
 *  Created on: 14.10.2016
 *      Author: kir_min
 */
#include "PushingSphere.cuh"

PushingSphere::PushingSphere(MDData *mdd, float R0, float vSphere, float4 centerPoint, int updatefreq, float sigma, float epsilon, const char* outdatfilename, int ljOrHarmonic, int* pushMask){
	printf("Initializing Pushing Sphere potential\n");
	this->mdd = mdd;
	this->R0 = R0;
	this->vSphere = vSphere;
	this->centerPoint = centerPoint;
	this->updatefreq = updatefreq;
	this->sigma = sigma;
	this->epsilon = epsilon;
	this->ljOrHarmonic = ljOrHarmonic;
	strcpy(filename, outdatfilename);

	printf("The sphere will change on %f nm every ps\n", vSphere);

	FILE* datout = fopen(filename, "w");
	fclose(datout); 
	
	this->blockCount = (mdd->N - 1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	
	h_mask = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_mask, mdd->N*sizeof(int));

	for(int i = 0; i < mdd->N; i++){
		h_mask[i] = pushMask[i];
	}
	cudaMemcpy(d_mask, h_mask, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	h_pressureOnSphere = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_pressureOnSphere, mdd->N*sizeof(float));

	cudaMemcpy(d_pressureOnSphere, h_pressureOnSphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);


	printf("Done initializing Pushing Sphere potential\n");
}

PushingSphere::~PushingSphere(){
	free(h_pressureOnSphere);
	cudaFree(d_pressureOnSphere);
}

__global__ void pushingSphereLJ_kernel(int* d_mask, float* d_pressureOnSphere, float R, float4 r0, float sigma, float epsilon){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){
		if(d_mask[d_i] == 1){
			float p = d_pressureOnSphere[d_i];
			float4 f = c_mdd.d_force[d_i];
			float4 ri = c_mdd.d_coord[d_i];

			float4 R_ri;
			R_ri.x = ri.x - r0.x;
			R_ri.y = ri.y - r0.y;
			R_ri.z = ri.z - r0.z;

			float R_ri2 = R_ri.x*R_ri.x + R_ri.y*R_ri.y + R_ri.z*R_ri.z;
			float mod_R_ri = sqrtf(R_ri2);

			float r = R - mod_R_ri;

			float sor = sigma/r;
			float sor2 = sor*sor;
			//float sor6 = sor2*sor2*sor2;
			//float mul = 6.0f*epsilon*sor6/r;
			float mul = 2.0f*epsilon*sor2/r;
		
			float4 df;
			df.x = -mul*R_ri.x/mod_R_ri;
			df.y = -mul*R_ri.y/mod_R_ri;
			df.z = -mul*R_ri.z/mod_R_ri;
		
			p += mul/(R*R);

			d_pressureOnSphere[d_i] = p; // p = force/r^2

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

__global__ void pushingSphereHarmonic_kernel(int* d_mask, float* d_pressureOnSphere, float R, float4 r0, float sigma, float epsilon){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){
		if(d_mask[d_i] == 1){
			float k = epsilon;
			float p = d_pressureOnSphere[d_i];
			float4 f = c_mdd.d_force[d_i];
			float4 ri = c_mdd.d_coord[d_i];

			float4 R_ri;
			R_ri.x = ri.x - r0.x;
			R_ri.y = ri.y - r0.y;
			R_ri.z = ri.z - r0.z;

			float R_ri2 = R_ri.x*R_ri.x + R_ri.y*R_ri.y + R_ri.z*R_ri.z;
			float mod_R_ri = sqrtf(R_ri2);

			float r = R - mod_R_ri;
			float mul;
			if(mod_R_ri > R){
				mul = k*r;
			}else{
				mul = 0.0;
			}

		
			float4 df;
			df.x = mul*R_ri.x/mod_R_ri;
			df.y = mul*R_ri.y/mod_R_ri;
			df.z = mul*R_ri.z/mod_R_ri;
		
			p += mul/(R*R);

			d_pressureOnSphere[d_i] = p; // p = force/r^2

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

void PushingSphere::compute(){
	this->radius = this->R0 - this->vSphere*mdd->step;

	if(ljOrHarmonic == PUSHING_SPHERE_LJ){
		pushingSphereLJ_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_pressureOnSphere, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}else
	if(ljOrHarmonic == PUSHING_SPHERE_HARMONIC){
		pushingSphereHarmonic_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_pressureOnSphere, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}

	if(mdd->step % this->updatefreq == 0){
		cudaMemcpy(h_pressureOnSphere, d_pressureOnSphere, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
		FILE* datout = fopen(filename, "a");
		float stress = 0.0;
		for(int i = 0; i < mdd->N; i++){
			stress += h_pressureOnSphere[i];
			h_pressureOnSphere[i] = 0;
		}
		float presureOnSphere = stress/(4.0*M_PI*this->updatefreq);
		fprintf(datout, "%d\t%f\t%e\n", mdd->step, this->radius, presureOnSphere);
		cudaMemcpy(d_pressureOnSphere, h_pressureOnSphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
		fclose(datout);
	}
}

__global__ void pushingSphereLJEnergy_kernel(int* d_mask, float* d_energy, float R, float4 r0, float sigma, float epsilon){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){
		if(d_mask[i] == 1){
			float4 ri = c_mdd.d_coord[i];

			float4 R_ri;
			R_ri.x = ri.x - r0.x;
			R_ri.y = ri.y - r0.y;
			R_ri.z = ri.z - r0.z;

			float R_ri2 = R_ri.x*R_ri.x + R_ri.y*R_ri.y + R_ri.z*R_ri.z;
			float mod_R_ri = sqrtf(R_ri2);
			float r = R - mod_R_ri;

			float sor = sigma/r;
			float sor2 = sor*sor;
			float energy = epsilon*sor2;
		
			d_energy[i] = energy;
		}
	}
}

__global__ void pushingSphereHarmonicEnergy_kernel(int* d_mask, float* d_energy, float R, float4 r0, float sigma, float epsilon){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){
		if(d_mask[i] == 1){
			float k = epsilon;
			float4 ri = c_mdd.d_coord[i];
		
			float4 R_ri;
			R_ri.x = ri.x - r0.x;
			R_ri.y = ri.y - r0.y;
			R_ri.z = ri.z - r0.z;

			float R_ri2 = R_ri.x*R_ri.x + R_ri.y*R_ri.y + R_ri.z*R_ri.z;
			float mod_R_ri = sqrtf(R_ri2);
			float r = R - mod_R_ri;
			float energy;
			if(mod_R_ri > R){
				energy = k*r*r/2.0;
			}else{
				energy = 0.0;
			}
		
			d_energy[i] = energy;
		}
	}
}

float PushingSphere::getEnergies(int energyId, int timestep){

	if(ljOrHarmonic == PUSHING_SPHERE_LJ){
		pushingSphereLJEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_energy, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}else
	if(ljOrHarmonic == PUSHING_SPHERE_HARMONIC){
		pushingSphereHarmonicEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_energy, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;
	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
