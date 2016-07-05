/*
 * Langevin.cu
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#include "Langevin.cuh"
#include "../Util/HybridTaus.cu"
#include "../md.cuh"

Langevin::Langevin(MDData *mdd){
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->mdd = mdd;
	gamma = 1.0f/getFloatParameter(PARAMETER_DAMPING);
	int seed = getIntegerParameter(PARAMETER_LANGEVIN_SEED);
	initRand(seed, mdd->N);
	float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
	var = sqrtf(2.0f*gamma*BOLTZMANN_CONSTANT*temperature/mdd->dt)/mdd->ftm2v;
	gamma /= mdd->ftm2v;
}

Langevin::~Langevin(){
	destroyRand();
}


__global__ void langevin_kernel(float gamma, float var, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 f = c_mdd.d_force[d_i];
		float4 v = c_mdd.d_vel[d_i];
		float m = c_mdd.d_mass[d_i];
		float4 rf = rforce(d_i);
		float mult = var*sqrtf(m);
		m = gamma*m; // /c_mdd.ftm2v;

		f.x += mult*rf.x - v.x*m;
		f.y += mult*rf.y - v.y*m;
		f.z += mult*rf.z - v.z*m;

		c_mdd.d_force[d_i] = f;
	}
}

void Langevin::compute(MDData *mdd){
	langevin_kernel<<<this->blockCount, this->blockSize>>>(gamma, var, mdd->N);
}

