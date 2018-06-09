/*
 * LeapFrogOverdumped.cu
 *
 * Created on: 11.07.2016
 *
 */

#include "LeapFrogOverdamped.cuh"
#include "../md.cuh"

LeapFrogOverdamped::LeapFrogOverdamped(MDData *mdd, float temperature, float gamma, float seed, int* h_fixAtoms){
	this->mdd = mdd;
	this->gamma = gamma;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;

	initRand(seed, mdd->N);

	cudaMalloc((void**)&d_fixAtoms, mdd->N*sizeof(int));
	cudaMemcpy(d_fixAtoms, h_fixAtoms, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	var = sqrtf(2.0f*BOLTZMANN_CONSTANT*temperature*gamma/mdd->dt);
}

LeapFrogOverdamped::~LeapFrogOverdamped(){
	cudaFree(d_fixAtoms);
}

__global__ void integrateLeapFrogOverdamped_kernel(float gamma, float var, int* d_fixAtoms){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < c_mdd.N){
		float4 coord = c_mdd.d_coord[i];
		int4 boxid = c_mdd.d_boxids[i];
		float4 f = c_mdd.d_force[i];
		float4 vel = c_mdd.d_vel[i];

		float tau = c_mdd.dt;

		float4 rf = rforce(i);
		float df = tau/gamma;

		vel.x = df*(f.x + var*rf.x);	// [nm]
		vel.y = df*(f.y + var*rf.y);	// [nm]
		vel.z = df*(f.z + var*rf.z);	// [nm]

		vel.w += gamma*(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)/(c_mdd.d_mass[i]*2.0f*tau);

		if (d_fixAtoms[i] == 0){
			coord.x += vel.x;
			coord.y += vel.y;
			coord.z += vel.z;
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

		c_mdd.d_vel[i] = vel;
		c_mdd.d_coord[i] = coord;
		c_mdd.d_force[i] = f;
		c_mdd.d_boxids[i] = boxid;
	}
}

void LeapFrogOverdamped::integrateStepOne(){
	//Do nothing
}

void LeapFrogOverdamped::integrateStepTwo(){
	integrateLeapFrogOverdamped_kernel<<<this->blockCount, this->blockSize>>>(gamma, var, d_fixAtoms);
}
