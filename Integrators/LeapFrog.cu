/*
 * LeapFrog.cu
 *
 *  Created on: 22.08.2012
 *      Author: zhmurov
 */

#include "LeapFrog.cuh"
#include "../md.cuh"

LeapFrog::LeapFrog(MDData *mdd){
	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;
}

LeapFrog::~LeapFrog(){

}

void LeapFrog::integrate_step_one (MDData *c_mdd){
	// Do nothing
}

__global__ void integrateLeapFrog_step_two_kernel(){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 vel = c_mdd.d_vel[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];
		m = 1.0f/m;
		m *= c_mdd.dt;
		m *= c_mdd.ftm2v;

		vel.x += m*f.x;
		vel.y += m*f.y;
		vel.z += m*f.z;

		/*vel.x *= 0.9f;
		vel.y *= 0.9f;
		vel.z *= 0.9f;*/
		vel.w += vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;

		coord.x += vel.x*c_mdd.dt;
		coord.y += vel.y*c_mdd.dt;
		coord.z += vel.z*c_mdd.dt;

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

		/*coord.x -= c_mdd.bc.rlo.x;
		coord.y -= c_mdd.bc.rlo.y;
		coord.z -= c_mdd.bc.rlo.z;

		coord.x -= floor(coord.x/c_mdd.bc.len.x)*c_mdd.bc.len.x;
		coord.y -= floor(coord.y/c_mdd.bc.len.y)*c_mdd.bc.len.y;
		coord.z -= floor(coord.z/c_mdd.bc.len.z)*c_mdd.bc.len.z;

		coord.x += c_mdd.bc.rlo.x;
		coord.y += c_mdd.bc.rlo.y;
		coord.z += c_mdd.bc.rlo.z;*/

		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;

		c_mdd.d_vel[d_i] = vel;
		c_mdd.d_coord[d_i] = coord;
		c_mdd.d_force[d_i] = f;
		c_mdd.d_boxids[d_i] = boxid;
	}
}

void LeapFrog::integrate_step_two (MDData *c_mdd){
	integrateLeapFrog_step_two_kernel<<<this->blockCount, this->blockSize>>>();
}


