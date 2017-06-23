/*
 * VelocityVerlet.cu
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */


#include "VelocityVerlet.cuh"
#include "../md.cuh"

VelocityVerlet::VelocityVerlet(MDData *mdd){
	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;
}

VelocityVerlet::~VelocityVerlet(){
}

__global__ void integrateVelocityVerletStepOne_kernel(float dt){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){

		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 v = c_mdd.d_vel[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];

		float mult = 0.5f*dt*c_mdd.ftm2v/m;

		v.x += mult*f.x;
		v.y += mult*f.y;
		v.z += mult*f.z;

//		v.x*=0.5f;
//		v.y*=0.5f;
//		v.z*=0.5f;

		coord.x += v.x*dt;
		coord.y += v.y*dt;
		coord.z += v.z*dt;

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

		/*r.x -= c_mdd.bc.rlo.x;
		r.y -= c_mdd.bc.rlo.y;
		r.z -= c_mdd.bc.rlo.z;

		r.x -= floor(r.x/c_mdd.bc.len.x)*c_mdd.bc.len.x;
		r.y -= floor(r.y/c_mdd.bc.len.y)*c_mdd.bc.len.y;
		r.z -= floor(r.z/c_mdd.bc.len.z)*c_mdd.bc.len.z;

		r.x += c_mdd.bc.rlo.x;
		r.y += c_mdd.bc.rlo.y;
		r.z += c_mdd.bc.rlo.z;*/

		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;

		c_mdd.d_vel[d_i] = v;
		c_mdd.d_coord[d_i] = coord;
		c_mdd.d_force[d_i] = f;
		c_mdd.d_boxids[d_i] = boxid;
	}
}


void VelocityVerlet::integrateStepOne(){
	integrateVelocityVerletStepOne_kernel<<<this->blockCount, this->blockSize>>>(dt);
}

__global__ void integrateVelocityVerletStepTwo_kernel(float dt){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){

		float4 v = c_mdd.d_vel[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];

		float mult = 0.5f*dt*c_mdd.ftm2v/m;

		v.x += mult*f.x;
		v.y += mult*f.y;
		v.z += mult*f.z;

		v.w += v.x*v.x + v.y*v.y + v.z*v.z;
//		v.w = v.x*v.x + v.y*v.y + v.z*v.z;

		c_mdd.d_vel[d_i] = v;
	}
}

void VelocityVerlet::integrateStepTwo(){
	integrateVelocityVerletStepTwo_kernel<<<this->blockCount, this->blockSize>>>(dt);
}

