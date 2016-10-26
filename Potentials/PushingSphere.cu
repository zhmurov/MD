/*
 * PushingSphere.cu
 *
 *  Created on: 14.10.2016
 *      Author: kir_min
 */
#include "PushingSphere.cuh"

PushingSphere::PushingSphere(MDData *mdd, float R0, float R, float4 centerPoint, int updatefreq, float sigma, float epsilon, const char* outdatfilename){
	printf("Initializing Pushing Sphere potential\n");
	this->mdd = mdd;
	this->R0 = R0;
	this->R = R;
	this->centerPoint = centerPoint;
	this->updatefreq = updatefreq;
	this->sigma = sigma;
	this->epsilon = epsilon;
	strcpy(filename, outdatfilename);

	FILE* datout = fopen(filename, "w");
	fclose(datout); 
	
	this->blockCount = (mdd->N - 1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	h_p_sphere = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_p_sphere, mdd->N*sizeof(float));

	cudaMemcpy(d_p_sphere, h_p_sphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
	printf("Done initializing Pushing Sphere potential\n");
}

PushingSphere::~PushingSphere(){
	free(h_p_sphere);
	cudaFree(d_p_sphere);
}

__global__ void pushSphereCompute_kernel(float* d_p_sphere, float R, float4 r0, float sigma, float epsilon){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){
		if(c_mdd.d_atomTypes[d_i] == 0){	//FOR DNA
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
		
			p_sphere += mul;

			d_p_sphere[d_i] = p_sphere/(R*R);

			f.x += df.x;
			f.y += df.y;
			f.z += df.z;

			c_mdd.d_force[d_i] = f;
		}	
	}
}

void PushingSphere::compute(){
	float alpha = (float)mdd->step/(float)mdd->numsteps;
	float radius = this->R0*(1.0f - alpha) + this->R*alpha;
	pushSphereCompute_kernel<<<this->blockCount, this->blockSize>>>(d_p_sphere, radius, this->centerPoint, this->sigma, this->epsilon);
	if(mdd->step % this->updatefreq == 0){
		cudaMemcpy(h_p_sphere, d_p_sphere, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
		FILE* datout = fopen(filename, "a");
		float stress = 0.0;
		for(int i = 0; i < mdd->N; i++){
			stress += h_p_sphere[i];
			h_p_sphere[i] = 0;
		}
		float presureOnSphere = stress/(4.0*M_PI*this->updatefreq);
		fprintf(datout, "%d\t%f\t%e\n", mdd->step, radius, presureOnSphere);
		cudaMemcpy(d_p_sphere, h_p_sphere, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
		fclose(datout);
	}
}
float PushingSphere::get_energies(int energy_id, int timestep){
	return 0;
}
