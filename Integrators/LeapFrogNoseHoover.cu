/*
 * LeapFrogNoseHoover.cu
 *
 *  Created on: 19.12.2013
 *      Author: zhmurov
 */

#include "LeapFrogNoseHoover.cuh"
#include "../md.cuh"

LeapFrogNoseHoover::LeapFrogNoseHoover(MDData *mdd, float tau, float T0){
	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;
	h_T = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_T, mdd->N*sizeof(float));
	gamma = 0.0f;
	this->tau = tau;
	this->T0 = T0;
	printf("mdd->N = %d\n", mdd->N);
	reduction = new Reduction(mdd->N);
}

LeapFrogNoseHoover::~LeapFrogNoseHoover(){

}

void LeapFrogNoseHoover::integrate_step_one (MDData *c_mdd){
	// Do nothing
}

__global__ void integrateLeapFrogNoseHoover_step_two_kernel(float gamma, float* d_T){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < c_mdd.N){
		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 vel = c_mdd.d_vel[d_i];
		float4 f = c_mdd.d_force[d_i];
		float m = c_mdd.d_mass[d_i];
		float mult = 1.0f/m;
		mult *= c_mdd.dt;
		mult *= c_mdd.ftm2v;

		vel.x += mult*f.x - c_mdd.dt*gamma*vel.x;
		vel.y += mult*f.y - c_mdd.dt*gamma*vel.y;
		vel.z += mult*f.z - c_mdd.dt*gamma*vel.z;

		/*vel.x *= 0.9f;
		vel.y *= 0.9f;
		vel.z *= 0.9f;*/
		float temp = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
		d_T[d_i] = temp*m;
		vel.w += temp;

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

void LeapFrogNoseHoover::integrate_step_two (MDData *c_mdd){
	integrateLeapFrogNoseHoover_step_two_kernel<<<this->blockCount, this->blockSize>>>(gamma, d_T);
	/*cudaMemcpy(h_T, d_T, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float temp = 0.0f;
	int i;
	for(i = 0; i < mdd->N; i++){
		temp += h_T[i];
	}
	temp /= ((float)mdd->N)*3.0f*BOLTZMANN_CONSTANT;*/
	float temp = reduction->rsum(d_T);
	temp /= ((float)mdd->N)*3.0f*BOLTZMANN_CONSTANT;
	//printf("T=%f\t%f\t%f\n", temp, temp2, temp-temp2);

	gamma += mdd->dt*(1.0f/tau)*(1.0f - T0/temp);
}


