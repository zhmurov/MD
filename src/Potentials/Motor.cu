/*
 * Motor.cu
 *
 *  Created on: 24.07.2017
 *      Author: kir_min
 */
#include "Motor.cuh"

Motor::Motor(MDData *mdd, float R0, float4 centerPoint, float motorForce, float radiusHole, float h, int updatefreq, float sigma, float epsilon, const char* outdatfilename, int* pushMask){
	printf("Initializing Motor potential\n");
	this->mdd = mdd;
	this->R0 = R0;
	this->motorForce = motorForce;
	this->h = h;
	this->radiusHole = radiusHole;
	this->centerPoint = centerPoint;
	this->updatefreq = updatefreq;
	this->sigma = sigma;
	this->epsilon = epsilon;
	strcpy(filename, outdatfilename);

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


	printf("Done initializing Motor potential\n");
}

Motor::~Motor(){
	free(h_pressureOnSphere);
	free(h_mask);
	free(h_energy);
	cudaFree(d_pressureOnSphere);
	cudaFree(h_mask);
	cudaFree(h_energy);
}

__global__ void motor_kernel(int* d_mask, float* d_pressureOnSphere, float R, float4 r0, float motorForce, float radiusHole, float h, float sigma, float epsilon){
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

			float4 df;

			if((R_ri.x*R_ri.x + R_ri.z*R_ri.z <= h) && ((ri.y >= -R-h) || (R_ri.y <= -R+h))){
				df.x = 0.0;
				df.y = motorForce;
				df.z = 0.0;
			}else{
				float R_ri2 = R_ri.x*R_ri.x + R_ri.y*R_ri.y + R_ri.z*R_ri.z;
				float mod_R_ri = sqrtf(R_ri2);

				float r = R - mod_R_ri;

				float sor = sigma/r;
				float sor2 = sor*sor;
				//float sor6 = sor2*sor2*sor2;
				//float mul = 6.0f*epsilon*sor6/r;
				float mul = 2.0f*epsilon*sor2/r;
		
				df.x = -mul*R_ri.x/mod_R_ri;
				df.y = -mul*R_ri.y/mod_R_ri;
				df.z = -mul*R_ri.z/mod_R_ri;
		
				p += mul/(R*R);
			}

			d_pressureOnSphere[d_i] = p; // p = force/r^2

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

void Motor::compute(){
	motor_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_pressureOnSphere, this->R0, this->centerPoint, this->motorForce, this->radiusHole, this->h, this->sigma, this->epsilon);

	if(mdd->step % this->updatefreq == 0){
		cudaMemcpy(h_pressureOnSphere, d_pressureOnSphere, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
		FILE* datout = fopen(filename, "a");
		float stress = 0.0;
		for(int i = 0; i < mdd->N; i++){
			stress += h_pressureOnSphere[i];
			h_pressureOnSphere[i] = 0;
		}
		float presureOnSphere = stress/(4.0*M_PI*this->updatefreq);
		fprintf(datout, "%d\t%f\t%e\n", mdd->step, this->R0, presureOnSphere);
		cudaMemcpy(d_pressureOnSphere, h_pressureOnSphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
		fclose(datout);
	}
}

__global__ void motorEnergy_kernel(int* d_mask, float* d_energy, float R, float4 r0, float sigma, float epsilon){
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

float Motor::getEnergies(int energyId, int timestep){

	motorEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_mask, d_energy, this->R0, this->centerPoint, this->sigma, this->epsilon);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;
	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
