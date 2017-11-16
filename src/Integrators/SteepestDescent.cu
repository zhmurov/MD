/*
 * SteepestDescent.cuh
 *
 * Created on: 21.03.2017
 *
 */

#include "SteepestDescent.cuh"
#include "../md.cuh"

SteepestDescent::SteepestDescent(MDData *mdd, float temperature, float gamma, float seed, float maxForce, int* fixAtoms){
	this->mdd = mdd;
	this->gamma = gamma;

	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;

	initRand(seed, mdd->N);

	h_fixAtoms = (int*)calloc(mdd->N, sizeof(int));
	memcpy(h_fixAtoms, fixAtoms, mdd->N*sizeof(int));
	cudaMalloc((void**)&d_fixAtoms, mdd->N*sizeof(int));
	cudaMemcpy(d_fixAtoms, h_fixAtoms, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	var = sqrtf(2.0f*BOLTZMANN_CONSTANT*temperature*gamma/mdd->dt);
}

SteepestDescent::~SteepestDescent(){
	free(h_fixAtoms);
	cudaFree(d_fixAtoms);
}

__global__ void integrateSteepestDescent_kernel(float gamma, float var, float maxForce, int* d_fixAtoms){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < c_mdd.N){
		float4 coord = c_mdd.d_coord[i];
		int4 boxid = c_mdd.d_boxids[i];
		float4 f = c_mdd.d_force[i];
		float4 vel = c_mdd.d_vel[i];

		float tau = c_mdd.dt;

		float4 rf = rforce(i);
		float df = tau/gamma;

		float modF = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);

		if(modF > maxForce){
			f.x = f.x*maxForce/modF;
			f.y = f.y*maxForce/modF;
			f.z = f.z*maxForce/modF;
		}

		vel.x = df*(f.x + var*rf.x);
		vel.y = df*(f.y + var*rf.y);
		vel.z = df*(f.z + var*rf.z);

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

void SteepestDescent::integrateStepOne(){
	//Do nothing
}

void SteepestDescent::integrateStepTwo(){
	integrateSteepestDescent_kernel<<<this->blockCount, this->blockSize>>>(gamma, var, maxForce, d_fixAtoms);
}
