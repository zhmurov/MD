/*
 * Langevin.cu
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 *  Changes: 12.07.2016
 *	Author: kir_min
*/

#include "Langevin.cuh"
#include "../Util/HybridTaus.cu"
#include "../md.cuh"

Langevin::Langevin(MDData *mdd, float damping, int seed, float temperature){
	printf("Initializing langevin potential\n");
	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->mdd = mdd;
	this->damping = damping;
	this->temperature = temperature;
	initRand(seed, mdd->N);
	this->var = sqrtf(2.0f*damping*BOLTZMANN_CONSTANT*temperature/mdd->dt);
	printf("Done initializing langevin potential\n");
}

Langevin::~Langevin(){
	destroyRand();
}


__global__ void langevin_kernel(float damping, float var, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		float4 f = c_mdd.d_force[d_i];
		float4 v = c_mdd.d_vel[d_i];
		float m = c_mdd.d_mass[d_i];
		float4 rf = rforce(d_i);
		float mult = var*sqrtf(m);

		float gamma = damping*m;

		f.x += mult*rf.x - gamma*v.x;
		f.y += mult*rf.y - gamma*v.y;
		f.z += mult*rf.z - gamma*v.z;

		c_mdd.d_force[d_i] = f;
	}
}

void Langevin::compute(){
	langevin_kernel<<<this->blockCount, this->blockSize>>>(this->damping, this->var, mdd->N);
}
