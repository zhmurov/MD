/*
 * SteepestDescent.cuh
 *
 * Created on: 21.03.2017
 *
 */

#include "SteepestDescent.cuh"
#include "../md.cuh"

SteepestDescent::SteepestDescent(MDData *mdd, float T, float seed, int* h_fixatoms, float maxForce){

	this->mdd = mdd;
	this->dt = mdd->dt;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;

	this->maxForce = maxForce;

	cudaMalloc((void**)&d_fixatoms, mdd->N*sizeof(int));
	cudaMemcpy(d_fixatoms, h_fixatoms, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	initRand(seed, mdd->N);

	h_gama = (float*)calloc(mdd->N, sizeof(float));
	h_var = (float*)calloc(mdd->N, sizeof(float));

	cudaMalloc((void**)&d_gama, mdd->N*sizeof(float));
	cudaMalloc((void**)&d_var, mdd->N*sizeof(float));

	for (int i = 0; i < mdd->N; i++){

		h_gama[i] = 6.0*M_PI*600.0*0.38;
		h_var[i] = sqrtf(2.0*BOLTZMANN_CONSTANT*T*h_gama[i]/mdd->dt);
	}

	cudaMemcpy(d_gama, h_gama, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_var, h_var, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

SteepestDescent::~SteepestDescent(){

	free(h_fixatoms);
	free(h_gama);
	free(h_var);
	cudaFree(d_fixatoms);
	cudaFree(d_gama);
	cudaFree(d_var);
}

__global__ void integrateSteepestDescent_kernel(float* d_gama, float* d_var, int* d_fixatoms, float maxForce){

	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_i < c_mdd.N){

		float4 coord = c_mdd.d_coord[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float4 f = c_mdd.d_force[d_i];
		float4 vel = c_mdd.d_vel[d_i];

		float gama = d_gama[d_i];
		float var = d_var[d_i];

		float tau = c_mdd.dt;

		float4 rf = rforce(d_i);
		float df = tau/gama;

		float modF = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);

		if(modF > maxForce){
			f.x = f.x*maxForce/modF;
			f.y = f.y*maxForce/modF;
			f.z = f.z*maxForce/modF;
		}

		vel.x = df*(f.x + var*rf.x);
		vel.y = df*(f.y + var*rf.y);
		vel.z = df*(f.z + var*rf.z);
		
		vel.w += gama*(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)/(c_mdd.d_mass[d_i]*2.0*tau);

		if (d_fixatoms[d_i] == 0){
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

void SteepestDescent::integrate_step_one(){
	//Do nothing
}

void SteepestDescent::integrate_step_two(){

	integrateSteepestDescent_kernel<<<this->blockCount, this->blockSize>>>(d_gama, d_var, d_fixatoms, maxForce);
}
