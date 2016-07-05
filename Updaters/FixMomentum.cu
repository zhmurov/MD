/*
 * FixMomentum.cu
 *
 *  Created on: 02.04.2016
 *      Author: zhmurov
 */


#include "FixMomentum.cuh"

FixMomentum::FixMomentum(MDData *mdd, int frequence){
	this->frequence = frequence;
	this->mdd = mdd;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->blockCount = (mdd->N-1)/this->blockSize + 1;
	reduction = new Reduction(mdd->N);
	reduction4 = new ReductionFloat4(mdd->N);

	h_muwcoord = (float4*)calloc(mdd->widthTot, sizeof(float4));
	h_mvel = (float4*)calloc(mdd->widthTot, sizeof(float4));
	h_angmom = (float4*)calloc(mdd->widthTot, sizeof(float4));
	h_inertiax = (float4*)calloc(mdd->widthTot, sizeof(float4));
	h_inertiay = (float4*)calloc(mdd->widthTot, sizeof(float4));
	h_inertiaz = (float4*)calloc(mdd->widthTot, sizeof(float4));
	cudaMalloc((void**)&d_muwcoord, mdd->widthTot*sizeof(float4));
	cudaMalloc((void**)&d_mvel, mdd->widthTot*sizeof(float4));
	cudaMalloc((void**)&d_angmom, mdd->widthTot*sizeof(float4));
	cudaMalloc((void**)&d_inertiax, mdd->widthTot*sizeof(float4));	
	cudaMalloc((void**)&d_inertiay, mdd->widthTot*sizeof(float4));	
	cudaMalloc((void**)&d_inertiaz, mdd->widthTot*sizeof(float4));	
}

FixMomentum::~FixMomentum(){
	free(h_muwcoord);
	free(h_mvel);
	free(h_angmom);
	free(h_inertiax);
	free(h_inertiay);
	free(h_inertiaz);
	cudaFree(d_muwcoord);
	cudaFree(d_mvel);
	cudaFree(d_angmom);
	cudaFree(d_inertiax);
	cudaFree(d_inertiay);
	cudaFree(d_inertiaz);
}

__global__ void precomputeCM_kernel(float4* d_muwcoord, float4* d_mvel, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		//float4 ri = tex1Dfetch(t_coord, d_i);
		float4 coord = c_mdd.d_coord[d_i];
		float4 vel = c_mdd.d_vel[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float m = c_mdd.d_mass[d_i]; 
		float4 muwcoord;
		muwcoord.x = coord.x + c_mdd.bc.len.x*boxid.x;
		muwcoord.y = coord.y + c_mdd.bc.len.y*boxid.y;
		muwcoord.z = coord.z + c_mdd.bc.len.z*boxid.z;
 		muwcoord.x *= m;
 		muwcoord.y *= m;
 		muwcoord.z *= m;
		d_muwcoord[d_i] = muwcoord;		
		vel.x *= m;
		vel.y *= m;
		vel.z *= m;
		d_mvel[d_i] = vel;
	}
}

__global__ void precomputeMomentum_kernel(float4* d_angmom, float4 xcm, float4 vcm, float4* d_inertiax, float4* d_inertiay, float4* d_inertiaz, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		//float4 ri = tex1Dfetch(t_coord, d_i);
		//float4 coord = c_mdd.d_coord[d_i];
		float4 vel = c_mdd.d_vel[d_i];
		//int4 boxid = c_mdd.d_boxids[d_i];
		//float m = c_mdd.d_mass[d_i]; 
		/*float dx = coord.x + c_mdd.bc.len.x*boxid.x;
		float dy = coord.y + c_mdd.bc.len.y*boxid.y;
		float dz = coord.z + c_mdd.bc.len.z*boxid.z;
		dx -= xcm.x;
		dy -= xcm.y;
		dz -= xcm.z;*/

		// Translational
		vel.x -= vcm.x;
		vel.y -= vcm.y;
		vel.z -= vcm.z;

		/*float4 angmom;
		angmom.x = m*(dy*vel.z - dz*vel.y);
		angmom.y = m*(dz*vel.x - dx*vel.z);
		angmom.z = m*(dx*vel.y - dy*vel.x);
		d_angmom[d_i] = angmom;
		float4 inertia;
		inertia.x = m*(dy*dy + dz*dz);
		inertia.y = -m*dx*dy;
		inertia.z = -m*dx*dz;
		d_inertiax[d_i] = inertia;
		inertia.x = -m*dx*dy;
		inertia.y = m*(dx*dx + dz*dz);
		inertia.z = -m*dy*dz;
		d_inertiay[d_i] = inertia;
		inertia.x = -m*dx*dz;
		inertia.y = -m*dy*dz;
		inertia.z = m*(dx*dx + dy*dy);
		d_inertiaz[d_i] = inertia;*/

		c_mdd.d_vel[d_i] = vel;
	}
}

__global__ void fixMomentum_kernel(float4 xcm, float4 vcm, float4 omega, int N){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < N){
		//float4 ri = tex1Dfetch(t_coord, d_i);
		float4 coord = c_mdd.d_coord[d_i];
		float4 vel = c_mdd.d_vel[d_i];
		int4 boxid = c_mdd.d_boxids[d_i];
		float dx = coord.x + c_mdd.bc.len.x*boxid.x;
		float dy = coord.y + c_mdd.bc.len.y*boxid.y;
		float dz = coord.z + c_mdd.bc.len.z*boxid.z;
		dx -= xcm.x;
		dy -= xcm.y;
		dz -= xcm.z;
		
		// Rotational
		vel.x -= omega.y*dz - omega.z*dy;
		vel.y -= omega.z*dx - omega.x*dz;
		vel.z -= omega.x*dy - omega.y*dx;

		c_mdd.d_vel[d_i] = vel;		
	}
}

inline void FixMomentum::update(MDData *mdd){
	precomputeCM_kernel<<<this->blockCount, this->blockSize>>>(d_muwcoord, d_mvel, mdd->N);
	float4 xcm = reduction4->rsum(d_muwcoord);
	xcm.x /= mdd->M;
	xcm.y /= mdd->M;
	xcm.z /= mdd->M;
	float4 vcm = reduction4->rsum(d_mvel);
	vcm.x /= mdd->M;
	vcm.y /= mdd->M;
	vcm.z /= mdd->M;
	precomputeMomentum_kernel<<<this->blockCount, this->blockSize>>>(d_angmom, xcm, vcm, d_inertiax, d_inertiay, d_inertiaz, mdd->N);
	/*float4 angmom = reduction4->rsum(d_angmom);
	float4 ix = reduction4->rsum(d_inertiax);
	float4 iy = reduction4->rsum(d_inertiay);
	float4 iz = reduction4->rsum(d_inertiaz);

	double inertia[3][3];
	double inverse[3][3];
	inertia[0][0] = ix.x; 	inertia[0][1] = ix.y; 	inertia[0][2] = ix.z;
	inertia[1][0] = iy.x; 	inertia[1][1] = iy.y; 	inertia[1][2] = iy.z;
	inertia[2][0] = iz.x; 	inertia[2][1] = iz.y; 	inertia[2][2] = iz.z;

	inverse[0][0] = inertia[1][1]*inertia[2][2] - inertia[1][2]*inertia[2][1];
	inverse[0][1] = -(inertia[0][1]*inertia[2][2] - inertia[0][2]*inertia[2][1]);
	inverse[0][2] = inertia[0][1]*inertia[1][2] - inertia[0][2]*inertia[1][1];

	inverse[1][0] = -(inertia[1][0]*inertia[2][2] - inertia[1][2]*inertia[2][0]);
	inverse[1][1] = inertia[0][0]*inertia[2][2] - inertia[0][2]*inertia[2][0];
	inverse[1][2] = -(inertia[0][0]*inertia[1][2] - inertia[0][2]*inertia[1][0]);
	
	inverse[2][0] = inertia[1][0]*inertia[2][1] - inertia[1][1]*inertia[2][0];
	inverse[2][1] = -(inertia[0][0]*inertia[2][1] - inertia[0][1]*inertia[2][0]);
	inverse[2][2] = inertia[0][0]*inertia[1][1] - inertia[0][1]*inertia[1][0];

	double determinant = inertia[0][0]*inertia[1][1]*inertia[2][2] +
		inertia[0][1]*inertia[1][2]*inertia[2][0] +
		inertia[0][2]*inertia[1][0]*inertia[2][1] -
		inertia[0][0]*inertia[1][2]*inertia[2][1] -
		inertia[0][1]*inertia[1][0]*inertia[2][2] -
		inertia[2][0]*inertia[1][1]*inertia[0][2];

	if(determinant > 0.0){
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				inverse[i][j] /= determinant;
			}
		}
	}

	float4 omega;
	omega.x = inverse[0][0]*angmom.x + inverse[0][1]*angmom.y + inverse[0][2]*angmom.z;
	omega.y = inverse[1][0]*angmom.x + inverse[1][1]*angmom.y + inverse[1][2]*angmom.z;
	omega.z = inverse[2][0]*angmom.x + inverse[2][1]*angmom.y + inverse[2][2]*angmom.z;
	fixMomentum_kernel<<<this->blockCount, this->blockSize>>>(xcm, vcm, omega, mdd->N);

	printf("vcm = %e, %e, %e;\t", vcm.x, vcm.y, vcm.z);
	printf("omega = %e, %e, %e\n", omega.x, omega.y, omega.z);*/
}
