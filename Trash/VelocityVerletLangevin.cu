/*
 * VelocityVerletLangevin.cu
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */


#include "../md.cuh"
#include "VelocityVerletLangevin.cuh"
#include "../Util/HybridTaus.cu"

VelocityVerletLangevin::VelocityVerletLangevin(MDData *mdd){
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = mdd->N/this->blockSize + 1;
	initRand(mdd->seed, mdd->N);
	gamma = 1.0f/getFloatParameter(PARAMETER_DAMPING);
	T = getFloatParameter(PARAMETER_TEMPERATURE);
	var = sqrtf(2.0f*gamma*BOLTZMANN_CONSTANT*T/dt);
	h_oldForces = (float4*)calloc(mdd->N, sizeof(float4));
	cudaMalloc((void**)&d_oldForces, mdd->N*sizeof(float4));
	cudaMemcpy(d_oldForces, h_oldForces, mdd->N*sizeof(float4), cudaMemcpyHostToDevice);
}

VelocityVerletLangevin::~VelocityVerletLangevin(){
	free(h_oldForces);
	cudaFree(d_oldForces);
}

void VelocityVerletLangevin::integrate_step_one (MDData *mdd){
	//integrateVelocityVerletLangevin_step_one_kernel<<<this->blockCount, this->blockSize>>>(dt, gamma, var, d_oldForces);
}

__global__ void integrateVelocityVerletLangevin_step_two_kernel(float dt, float gamma, float var, float4* d_oldForces){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 r = c_mdd.d_coord[d_i];
		float4 v = c_mdd.d_vel[d_i];
		float m = c_mdd.d_mass[d_i];
		float4 fn = d_oldForces[d_i];

		float m1 = (1.0f - 0.5f*dt*gamma);
		float m2 = 0.5f*dt*dt/m;

		r.x = r.x + dt*v.x*m1 + m2*fn.x;
		r.y = r.y + dt*v.y*m1 + m2*fn.y;
		r.z = r.z + dt*v.z*m1 + m2*fn.z;

		float4 fnp1 = c_mdd.d_force[d_i];
		/*float4 rf = rforce(d_i);

		var *= sqrtf(m);

		fnp1.x += var*rf.x;
		fnp1.y += var*rf.y;
		fnp1.z += var*rf.z;*/

		d_oldForces[d_i] = fnp1;

		float m3 = 1.0f/(1.0f + 0.5f*dt*gamma);

		v.x = m3*(v.x*m1 + m2*(fn.x + fnp1.x));
		v.y = m3*(v.y*m1 + m2*(fn.y + fnp1.y));
		v.z = m3*(v.z*m1 + m2*(fn.z + fnp1.z));

		v.w += v.x*v.x + v.y*v.y + v.z*v.z;

		if(r.x > c_mdd.bc.rhi.x){
			r.x -= c_mdd.bc.len.x;
		} else
		if(r.x < c_mdd.bc.rlo.x){
			r.x += c_mdd.bc.len.x;
		}
		if(r.y > c_mdd.bc.rhi.y){
			r.y -= c_mdd.bc.len.y;
		} else
		if(r.y < c_mdd.bc.rlo.y){
			r.y += c_mdd.bc.len.y;
		}
		if(r.z > c_mdd.bc.rhi.z){
			r.z -= c_mdd.bc.len.z;
		} else
		if(r.z < c_mdd.bc.rlo.z){
			r.z += c_mdd.bc.len.z;
		}

		fnp1.x = 0.0f;
		fnp1.y = 0.0f;
		fnp1.z = 0.0f;

		c_mdd.d_vel[d_i] = v;
		c_mdd.d_coord[d_i] = r;
		c_mdd.d_force[d_i] = fnp1;
	}
}

void VelocityVerletLangevin::integrate_step_two (MDData *mdd){
	integrateVelocityVerletLangevin_step_two_kernel<<<this->blockCount, this->blockSize>>>(dt, gamma, var, d_oldForces);
}

