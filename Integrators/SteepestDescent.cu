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

		/*vel.x = df*f.x;
		vel.y = df*f.y;
		vel.z = df*f.z;*/

		vel.x = df*(f.x + var*rf.x);
		vel.y = df*(f.y + var*rf.y);
		vel.z = df*(f.z + var*rf.z);
		
		float modF = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;

		vel.w += gama*modF/(c_mdd.d_mass[d_i]*2.0*tau);

		if(modF > maxForce){
			vel.x *= modF/maxForce;
			vel.y *= modF/maxForce;
			vel.z *= modF/maxForce;
		}

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

/*
	cudaMemcpy(mdd->h_coord, mdd->d_coord, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);

	int i, j;
	float3 rij;
	float rij_mod, r0, dev;		//deviation
	for (i = 0; i < (mdd->N - 1); i++){

		j = i+1;
		if ((i == topdata->bonds[i].i) && (j == topdata->bonds[i].j)){
			rij.x = mdd->h_coord[j].x*10.0 - mdd->h_coord[i].x*10.0;
			rij.y = mdd->h_coord[j].y*10.0 - mdd->h_coord[i].y*10.0;
			rij.z = mdd->h_coord[j].z*10.0 - mdd->h_coord[i].z*10.0;

			rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
			r0 = topdata->bonds[i].c0;
			dev = abs(rij_mod - r0);

			//printf("[%2d]-[%2d]\trij_mod = %5.5f\tr0 = %5.5f\tabs(dev) = %5.5f\t[ANGSTR]\n", i, (i+1), rij_mod, r0, dev);

			if (dev >= 2.0){
				printf("WARNING-WARNING-WARNING-WARNING-WARNING\n");
			}
		}
	}
*/
}
