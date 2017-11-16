/*
 * LeapFrogLangevin.cuh
 *
 *  Created on: 27.06.2017
 *      Author: zhmurov
	Author: ilya_kir
	Author: kir_min
 */

#include "LeapFrogLangevin.cuh"
#include "../md.cuh"

LeapFrogLangevin::LeapFrogLangevin(MDData *mdd, float T, float seed, int* h_fixAtoms, float damping){

	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;

	cudaMalloc((void**)&d_fixAtoms, mdd->N*sizeof(int));
	cudaMemcpy(d_fixAtoms, h_fixAtoms, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	initRand(seed, mdd->N);

	h_gamma = (float*)calloc(mdd->N, sizeof(float));
	h_var = (float*)calloc(mdd->N, sizeof(float));

	cudaMalloc((void**)&d_gamma, mdd->N*sizeof(float));
	cudaMalloc((void**)&d_var, mdd->N*sizeof(float));
	
	for (int i = 0; i < mdd->N; i++){
		if(damping < 0){
			h_gamma[i] = 6.0*M_PI*600.0*0.38;
			h_var[i] = sqrtf(2.0*BOLTZMANN_CONSTANT*T*h_gamma[i]/mdd->dt);
		}else{
			h_gamma[i] = damping*mdd->h_mass[i];
			h_var[i] = sqrtf(2.0*BOLTZMANN_CONSTANT*T*h_gamma[i]/mdd->dt);
		}
	}

	cudaMemcpy(d_gamma, h_gamma, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_var, h_var, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

LeapFrogLangevin::~LeapFrogLangevin(){

	free(h_gamma);
	free(h_var);
	free(h_fixAtoms);
	cudaFree(d_gamma);
	cudaFree(d_var);
	cudaFree(h_fixAtoms);
}

__global__ void integrateLeapFrogLangevin_kernel(float* d_gamma, float* d_var, int* d_fixAtoms){

	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];
		float4 vel = c_mdd.d_vel[d_i];

		float gamma = d_gamma[d_i];
		float var = d_var[d_i];

		float tau = c_mdd.dt;

		float4 rf = rforce(d_i);
		float obratTau = 1.0f/tau;
		float gammaM = gamma/(2.0f*m);

		vel.x = ((f.x+ var*rf.x)/m + vel.x*(obratTau - gammaM)) / (obratTau + gammaM);
		vel.y = ((f.y+ var*rf.y)/m + vel.y*(obratTau - gammaM)) / (obratTau + gammaM);
		vel.z = ((f.z+ var*rf.z)/m + vel.z*(obratTau - gammaM)) / (obratTau + gammaM);

		vel.w += vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;

		if(d_fixAtoms[d_i] == 0){
			coord.x += vel.x*tau;
			coord.y += vel.y*tau;
			coord.z += vel.z*tau;
		}

		if(coord.x > c_mdd.bc.rhi.x){
			coord.x -= c_mdd.bc.len.x;
			boxid.x ++;
		} else
		if(coord.x < c_mdd.bc.rlo.x){
			coord.x += c_mdd.bc.len.x;
			boxid.x --;
		}
		if(coord.y > c_mdd.bc.rhi.y){
			coord.y -= c_mdd.bc.len.y;
			boxid.y ++;
		} else
		if(coord.y < c_mdd.bc.rlo.y){
			coord.y += c_mdd.bc.len.y;
			boxid.y --;
		}
		if(coord.z > c_mdd.bc.rhi.z){
			coord.z -= c_mdd.bc.len.z;
			boxid.z ++;
		} else
		if(coord.z < c_mdd.bc.rlo.z){
			coord.z += c_mdd.bc.len.z;
			boxid.z --;
		}

		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;

		c_mdd.d_vel[d_i] = vel;
		c_mdd.d_coord[d_i] = coord;
		c_mdd.d_force[d_i] = f;
		c_mdd.d_boxids[d_i] = boxid;
	}
}


void LeapFrogLangevin::integrateStepOne (){
	// Do nothing
}

void LeapFrogLangevin::integrateStepTwo (){

	integrateLeapFrogLangevin_kernel<<<this->blockCount, this->blockSize>>>(d_gamma, d_var, d_fixAtoms);
}
