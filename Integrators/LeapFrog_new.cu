/*
 * LeapFrog.cu
 *
 *  Created on: 22.08.2012
 *      Author: zhmurov
 */

#include "LeapFrog_new.cuh"
#include "../md.cuh"

LeapFrog_new::LeapFrog_new(MDData *mdd, float T, float seed){

	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;

	initRand(seed, mdd->N);

	h_gama = (float*)calloc(mdd->N, sizeof(float));
	h_var = (float*)calloc(mdd->N, sizeof(float));

	cudaMalloc((void**)&d_gama, mdd->N*sizeof(float));
	cudaMalloc((void**)&d_var, mdd->N*sizeof(float));

	for (int i = 0; i < mdd->N; i++){

		h_gama[i] = 6.0*M_PI*600.0*0.38;
		h_var[i] = sqrtf(2.0*BOLTZMANN_CONSTANT*T*h_gama[i]/mdd->dt);
	}//TODO

	cudaMemcpy(d_gama, h_gama, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_var, h_var, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

LeapFrog_new::~LeapFrog_new(){

	free(h_gama);
	free(h_var);
	cudaFree(d_gama);
	cudaFree(d_var);
}

__global__ void integrateLeapFrog_underdumped(float* d_gama, float* d_var){

	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];
		float4 vel = c_mdd.d_vel[d_i];

		float gama = d_gama[d_i];
		float var = d_var[d_i];

		float tau = c_mdd.dt;

//TODO TEMP LANGEVIN

		float4 rf = rforce(d_i);
		float df = tau/m;

		vel.x += df*(f.x - vel.x*gama + var*rf.x);
		vel.y += df*(f.y - vel.y*gama + var*rf.y);
		vel.z += df*(f.z - vel.z*gama + var*rf.z);

		vel.w += vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;

		coord.x += vel.x*tau;
		coord.y += vel.y*tau;
		coord.z += vel.z*tau;

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


void LeapFrog_new::integrate_step_one (){
	// Do nothing
}

void LeapFrog_new::integrate_step_two (){

	integrateLeapFrog_underdumped<<<this->blockCount, this->blockSize>>>(d_gama, d_var);
}
