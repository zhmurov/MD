/*
 * BondsClass2Atom.cu
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 *  Changes: 15.08.2016
 *	Author: kir_min
 */

#include "BondsClass2Atom.cuh"

BondsClass2Atom::BondsClass2Atom(MDData *mdd, int bondCountPar, int bondCountTop, int4* pair, float4* bondCoeffs){
	printf("Initializing BondsClass2Atom potential\n");
	this->mdd = mdd;
	int i;
	h_bondPar = (float4*)calloc(bondCountPar, sizeof(float4));
	cudaMalloc((void**)&d_bondPar, bondCountPar*sizeof(float4));
	for(i = 0; i < bondCountPar; i++){
		h_bondPar[i].x = bondCoeffs[i].x;
		h_bondPar[i].y = 2.0f*bondCoeffs[i].y;
		h_bondPar[i].z = 3.0f*bondCoeffs[i].z;
		h_bondPar[i].w = 4.0f*bondCoeffs[i].w;
	}

	cudaMemcpy(d_bondPar, h_bondPar, bondCountPar*sizeof(float4), cudaMemcpyHostToDevice);

	// Count bonds per atom

	h_bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bondCount, mdd->N*sizeof(int));

	for (i = 0; i < bondCountTop; i++) {
		h_bondCount[pair[i].x]++;
		h_bondCount[pair[i].y]++;
	}

	// Calculate maxBond and lastBonded

	int maxBonds = 0;
	int lastBonded = 0;
	for(i = 0; i < mdd->N; i++){
		if(h_bondCount[i] > 0){
			lastBonded = i;
		}
		if(maxBonds < h_bondCount[i]){
			maxBonds = h_bondCount[i];
		}
	}

	bondedCount = lastBonded + 1;
	this->blockCount = (bondedCount-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	widthTot = ((bondedCount-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	h_bonds = (int2*)calloc(widthTot*maxBonds, sizeof(int2));
	cudaMalloc((void**)&d_bonds, widthTot*maxBonds*sizeof(int2));

	// Fill list of bonded neighbours for each atom

	for(i = 0; i < mdd->N; i++){
		h_bondCount[i] = 0;
	}
	int a1, a2, type;
	for (i = 0; i < bondCountTop; i++) {
		a1 = pair[i].x;
		a2 = pair[i].y;
		type = pair[i].z;
		h_bonds[h_bondCount[a1]*widthTot + a1].x = a2;
		h_bonds[h_bondCount[a2]*widthTot + a2].x = a1;
		h_bonds[h_bondCount[a1]*widthTot + a1].y = type;
		h_bonds[h_bondCount[a2]*widthTot + a2].y = type;
		h_bondCount[a1]++;
		h_bondCount[a2]++;
	}
	h_energies = (float*)calloc(bondedCount, sizeof(float));
	cudaMalloc((void**)&d_energies, bondedCount*sizeof(float));

	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bonds, h_bonds, widthTot*maxBonds*sizeof(int2), cudaMemcpyHostToDevice);
	printf("Done initializing BondsClass2Atom potential\n");
}

BondsClass2Atom::~BondsClass2Atom(){
	free(h_bondCount);
	free(h_bonds);
	free(h_bondPar);
	free(h_energies);
	cudaFree(d_bondCount);
	cudaFree(d_bonds);
	cudaFree(d_bondPar);
	cudaFree(d_energies);
}

__global__ void bondClass2_kernel(int* d_bondCount, int2* d_bonds, float4* d_bondPar, int widthTot, int bondedCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondedCount){
		int j;
		float4 f = c_mdd.d_force[d_i];
		float4 r1 = c_mdd.d_coord[d_i];
		float4 r2;
		for(j = 0; j < d_bondCount[d_i]; j++){
			int2 bond = d_bonds[j*widthTot + d_i];
			r2 = c_mdd.d_coord[bond.x];
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float4 par = d_bondPar[bond.y];
			r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;  // (r1-r2)^2
			r2.w = sqrtf(r2.w); // |r1-r2|
			//float r = r2.w;
			float rinv = 1.0f/r2.w;					// 1/|r1-r2|

			r2.w -= par.x;							// dr = |r1-r2|-r0
			float df = r2.w*(par.y + r2.w*(par.z + r2.w*par.w)); // (2*k2)*dr + (3*k3)*dr^2 + (4*k4)*dr^3
			df *= rinv;

			f.x += r2.x*df;
			f.y += r2.y*df;
			f.z += r2.z*df;

		}
		c_mdd.d_force[d_i] = f;
	}
}


void BondsClass2Atom::compute(){
	bondClass2_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bonds, d_bondPar, widthTot, bondedCount);
}

__global__ void bondClass2Energy_kernel(int* d_bondCount, int2* d_bonds, float4* d_bondPar, float* d_energies, int widthTot, int bondedCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondedCount){
		int j;
		float4 r1 = c_mdd.d_coord[d_i];
		float4 r2;
		float pot = 0.0f;
		for(j = 0; j < d_bondCount[d_i]; j++){
			int2 bond = d_bonds[j*widthTot + d_i];
			r2 = c_mdd.d_coord[bond.x];
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float4 par = d_bondPar[bond.y];
			r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
			r1.w = sqrtf(r2.w);
			r1.w -= par.x;
			r2.w = r1.w*r1.w;

			pot += (0.5f*par.y + (par.z/3.0f)*r1.w + 0.25f*par.w*r2.w)*r2.w;

		}
		d_energies[d_i] = pot;
	}
}

float BondsClass2Atom::getEnergies(int energyId, int timestep){
	bondClass2Energy_kernel<<<this->blockCount, this->blockSize>>>(d_bondCount, d_bonds, d_bondPar, d_energies, widthTot, bondedCount);
	//cudaThreadSynchronize();
	cudaMemcpy(h_energies, d_energies, bondedCount*sizeof(float), cudaMemcpyDeviceToHost);
	int i;
	double energy = 0.0f;
	for(i = 0; i < bondedCount; i++){
		energy += h_energies[i];
	}
	energy *= 0.5f;
	return energy;
}
