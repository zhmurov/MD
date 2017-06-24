/*
 * LeapFrogOverdumped.cu
 *
 * Created on: 11.07.2016
 *
 */

#include "LeapFrogOverdumped.cuh"
#include "../md.cuh"

LeapFrog_overdumped::LeapFrogOverdumped(MDData *mdd, float T, float seed, int* h_fixAtoms){

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

		h_gamma[i] = 6.0*M_PI*600.0*0.38;
		h_var[i] = sqrtf(2.0*BOLTZMANN_CONSTANT*T*h_gamma[i]/mdd->dt);
	}

	cudaMemcpy(d_gamma, h_gamma, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_var, h_var, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

LeapFrog_overdumped::~LeapFrogOverdumped(){

	free(h_fixAtoms);
	free(h_gamma);
	free(h_var);
	cudaFree(d_fixAtoms);
	cudaFree(d_gamma);
	cudaFree(d_var);
}

__global__ void integrateLeapFrogOverdumped_kernel(float* d_gamma, float* d_var, int* d_fixAtoms){

	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){

		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 f = c_mdd.d_force[d_i];
		float4 vel = c_mdd.d_vel[d_i];

		float gamma = d_gamma[d_i];
		float var = d_var[d_i];

		float tau = c_mdd.dt;

		float4 rf = rforce(d_i);
		float df = tau/gamma;

		vel.x = df*(f.x + var*rf.x);		// [nm]
		vel.y = df*(f.y + var*rf.y);		// [nm]
		vel.z = df*(f.z + var*rf.z);		// [nm]

		vel.w += gamma*(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)/(c_mdd.d_mass[d_i]*2.0*tau);

		// if d_i atom isn't fixed
		if (d_fixAtoms[d_i] == 0){
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

		c_mdd.d_vel[d_i] = vel;
		c_mdd.d_coord[d_i] = coord;
		c_mdd.d_force[d_i] = f;
		c_mdd.d_boxids[d_i] = boxid;
	}
}

void LeapFrogOverdumped::integrateStepOne(){
	//Do nothing
}

void LeapFrogOverdumped::integrateStepTwo(){
	integrateLeapFrogOverdumped_kernel<<<this->blockCount, this->blockSize>>>(d_gamma, d_var, d_fixAtoms);
}
