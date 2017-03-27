/*
 * PushingSphere.cu
 *
 *  Created on: 14.10.2016
 *      Author: kir_min
 */
#include "PushingSphere.cuh"

PushingSphere::PushingSphere(MDData *mdd, float R0, float R, float4 centerPoint, int updatefreq, float sigma, float epsilon, const char* outdatfilename, int lj_or_harmonic, int* push_mask){
	printf("Initializing Pushing Sphere potential\n");
	this->mdd = mdd;
	this->R0 = R0;
	this->R = R;
	this->centerPoint = centerPoint;
	this->updatefreq = updatefreq;
	this->sigma = sigma;
	this->epsilon = epsilon;
	this->lj_or_harmonic = lj_or_harmonic;
	strcpy(filename, outdatfilename);

	FILE* datout = fopen(filename, "w");
	fclose(datout); 
	
	this->blockCount = (mdd->N - 1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	
	h_mask = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_mask, mdd->N*sizeof(int));

	for(int i = 0; i < mdd->N; i++){
		h_mask[i] = push_mask[i];
	}
	cudaMemcpy(d_mask, h_mask, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	h_p_sphere = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_p_sphere, mdd->N*sizeof(float));

	cudaMemcpy(d_p_sphere, h_p_sphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);

	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));

	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);


	printf("Done initializing Pushing Sphere potential\n");
}

PushingSphere::~PushingSphere(){
	free(h_p_sphere);
	cudaFree(d_p_sphere);
}

__global__ void PushingSphereLJ_Compute_kernel(int* d_mask, float* d_p_sphere, float R, float4 r0, float sigma, float epsilon){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){
		if(d_mask[d_i] == 1){
			float p_sphere = d_p_sphere[d_i];
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
		
			p_sphere += mul/(R*R);

			d_p_sphere[d_i] = p_sphere;

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

__global__ void PushingSphereHarmonic_Compute_kernel(int* d_mask, float* d_p_sphere, float R, float4 r0, float sigma, float epsilon){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){
		if(d_mask[d_i] == 1){
			float k = epsilon;
			float p_sphere = d_p_sphere[d_i];
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
		
			p_sphere += mul/(R*R);

			d_p_sphere[d_i] = p_sphere;

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

void PushingSphere::compute(){
	float alpha = (float)mdd->step/(float)mdd->numsteps;
	this->radius = this->R0*(1.0f - alpha) + this->R*alpha;

	if(lj_or_harmonic == PUSHING_SPHERE_LJ){
		PushingSphereLJ_Compute_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_p_sphere, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}
	if(lj_or_harmonic == PUSHING_SPHERE_HARMONIC){
		PushingSphereHarmonic_Compute_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_p_sphere, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}

	if(mdd->step % this->updatefreq == 0){
		cudaMemcpy(h_p_sphere, d_p_sphere, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
		FILE* datout = fopen(filename, "a");
		float stress = 0.0;
		for(int i = 0; i < mdd->N; i++){
			stress += h_p_sphere[i];
			h_p_sphere[i] = 0;
		}
		float presureOnSphere = stress/(4.0*M_PI*this->updatefreq);
		fprintf(datout, "%d\t%f\t%e\n", mdd->step, this->radius, presureOnSphere);
		cudaMemcpy(d_p_sphere, h_p_sphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
		fclose(datout);
	}
}

__global__ void PushingSphereLJ_Energy_kernel(int* d_mask, float* d_energy, float R, float4 r0, float sigma, float epsilon){
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

__global__ void PushingSphereHarmonic_Energy_kernel(int* d_mask, float* d_energy, float R, float4 r0, float sigma, float epsilon){
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

float PushingSphere::get_energies(int energy_id, int timestep){

	if(lj_or_harmonic == PUSHING_SPHERE_LJ){
		PushingSphereLJ_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_energy, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}
	if(lj_or_harmonic == PUSHING_SPHERE_HARMONIC){
		PushingSphereHarmonic_Energy_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_energy, this->radius, this->centerPoint, this->sigma, this->epsilon);
	}

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;
	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
