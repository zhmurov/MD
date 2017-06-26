/*
 * BondClass2Pair.cu
 *
 *  Created on: 10.09.2012
 *      Author: zhmurov
 */

#include "BondsClass2Pair.cuh"



BondsClass2Pair::BondsClass2Pair(MDData *mdd, std::vector<int3> &bonds, std::vector<Coeffs> &parameters){
	this->mdd = mdd;
	int i;

	// Coefficents

	int btCount = parameters.size();
	h_bondPar = (float4*)calloc(btCount, sizeof(float4));
	cudaMalloc((void**)&d_bondPar, btCount*sizeof(float4));
	for (i=0; i!=parameters.size(); i++) {
		Coeffs& coeffs = parameters[i];
		h_bondPar[i].x = atof(coeffs.coeffs.at(0).c_str()); // l0
		h_bondPar[i].y = 2.0f*atof(coeffs.coeffs.at(1).c_str()); // 2*k2
		h_bondPar[i].z = 3.0f*atof(coeffs.coeffs.at(2).c_str()); // 3*k3
		h_bondPar[i].w = 4.0f*atof(coeffs.coeffs.at(3).c_str()); // 4*k4
	}

	cudaMemcpy(d_bondPar, h_bondPar, btCount*sizeof(float4), cudaMemcpyHostToDevice);

	// Count bonds per atom

	h_bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bondCount, mdd->N*sizeof(int));
	bondCount = bonds.size();
	for (i=0;i!=bonds.size();i++) {
		h_bondCount[bonds[i].x]++;
		h_bondCount[bonds[i].y]++;
	}

	// Calculate maxBond and lastBonded

	for(i = 0; i < mdd->N; i++){
		if(h_bondCount[i] > 0){
			lastBonded = i;
		}
		if(maxBonds < h_bondCount[i]){
			maxBonds = h_bondCount[i];
		}
	}

	this->blockCount = (bondCount-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	blockCountSum = lastBonded/DEFAULT_BLOCK_SIZE + 1;
	blockSizeSum = DEFAULT_BLOCK_SIZE;
	widthTot = (lastBonded/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;

	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);

	h_bonds = (int3*)calloc(bondCount, sizeof(int3));
	cudaMalloc((void**)&d_bonds, bondCount*sizeof(int3));
	h_refs = (int2*)calloc(bondCount, sizeof(int2));
	cudaMalloc((void**)&d_refs, bondCount*sizeof(int2));

	// Fill list of bonded neighbours for each atom

	for(i = 0; i < mdd->N; i++){
		h_bondCount[i] = 0;
	}
	int a1, a2, type;
	for (i=0;i!=bonds.size();i++) {
		a1 = bonds[i].x;
		a2 = bonds[i].y;
		type = bonds[i].z;
		h_bonds[i].x = a1;
		h_bonds[i].y = a2;
		h_bonds[i].z = type;
		h_refs[i].x = h_bondCount[a1]*widthTot + a1;
		h_refs[i].y = h_bondCount[a2]*widthTot + a2;
		h_bondCount[a1]++;
		h_bondCount[a2]++;
	}

	h_forces = (float4*)calloc(maxBonds*widthTot, sizeof(float4));
	cudaMalloc((void**)&d_forces, maxBonds*widthTot*sizeof(float4));

	h_energies = (float*)calloc(bondCount, sizeof(float));
	cudaMalloc((void**)&d_energies, bondCount*sizeof(float));

	cudaMemcpy(d_bondCount, h_bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bonds, h_bonds, bondCount*sizeof(int3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_refs, h_refs, bondCount*sizeof(int2), cudaMemcpyHostToDevice);
}

BondsClass2Pair::~BondsClass2Pair(){

	free(h_bonds);
	free(h_refs);
	free(h_bondPar);
	free(h_bondCount);
	free(h_forces);
	free(h_energies);

	cudaFree(d_bonds);
	cudaFree(d_refs);
	cudaFree(d_bondPar);
	cudaFree(d_bondCount);
	cudaFree(d_forces);
	cudaFree(d_energies);
}

__global__ void bondClass2Pair_kernel(int3* d_bonds, int2* d_refs, float4* d_bondPar, float4* d_forces, int bondCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondCount){
		int3 bond = d_bonds[d_i];
		float4 r1 = c_mdd.d_coord[bond.x];
		float4 r2 = c_mdd.d_coord[bond.y];
		float4 par = d_bondPar[bond.z];
		int2 ref = d_refs[d_i];

		r2.x -= r1.x;
		r2.y -= r1.y;
		r2.z -= r1.z;

		float3 pb = c_mdd.bc.len;
		r2.x -= rint(r2.x/pb.x)*pb.x;
		r2.y -= rint(r2.y/pb.y)*pb.y;
		r2.z -= rint(r2.z/pb.z)*pb.z;

		r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;  // (r1-r2)^2
		r2.w = sqrtf(r2.w);						// |r1-r2|
		float rinv = 1.0f/r2.w;					// 1/|r1-r2|

		r2.w -= par.x;							// dr = |r1-r2|-r0
		float df = r2.w*(par.y + r2.w*(par.z + r2.w*par.w)); // (2*k2)*dr + (3*k3)*dr^2 + (4*k4)*dr^3
		df *= rinv;

		r1.x = r2.x*df;
		r1.y = r2.y*df;
		r1.z = r2.z*df;

		d_forces[ref.x] = r1;

		r1.x = -r1.x;
		r1.y = -r1.y;
		r1.z = -r1.z;

		d_forces[ref.y] = r1;
	}
}

__global__ void bondClass2PairSum_kernel(int* d_count, float4* d_forces, int widthTot, int lastBonded){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i <= lastBonded){
		float4 f = c_mdd.d_force[d_i];
		int j;
		for(j = 0; j < d_count[d_i]; j++){
			float4 df = d_forces[j*widthTot + d_i];
			f.x += df.x;
			f.y += df.y;
			f.z += df.z;
		}
		c_mdd.d_force[d_i] = f;
	}
}


void BondsClass2Pair::compute(MDData *mdd){
	bondClass2Pair_kernel<<<this->blockCount, this->blockSize>>>(d_bonds, d_refs, d_bondPar, d_forces, bondCount);
	bondClass2PairSum_kernel<<<blockCountSum, blockSizeSum>>>(d_bondCount, d_forces, widthTot, lastBonded);
}

__global__ void bondClass2PairEnergy_kernel(int3* d_bonds, float4* d_bondPar, float* d_energies, int bondCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondCount){
		int3 bond = d_bonds[d_i];
		float4 r1 = c_mdd.d_coord[bond.x];
		float4 r2 = c_mdd.d_coord[bond.y];
		float4 par = d_bondPar[bond.z];

		r2.x -= r1.x;
		r2.y -= r1.y;
		r2.z -= r1.z;

		float3 pb = c_mdd.bc.len;
		r2.x -= rint(r2.x/pb.x)*pb.x;
		r2.y -= rint(r2.y/pb.y)*pb.y;
		r2.z -= rint(r2.z/pb.z)*pb.z;

		r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;  // (r1-r2)^2
		r1.w = sqrtf(r2.w);						// |r1-r2|
		r1.w -= par.x;							// dr = |r1-r2|-r0
		r2.w = r1.w*r1.w;						// dr^2

		float pot = (0.5f*par.y + (par.z/3.0f)*r1.w + 0.25f*par.w*r2.w)*r2.w; // k2*dr^2 + k3*dr^3 + k4*dr^4

		d_energies[d_i] = pot;

	}
}

float BondsClass2Pair::getEnergies(int energyId, int timestep){
	bondClass2PairEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_bonds, d_bondPar, d_energies, bondCount);
	//cudaThreadSynchronize();
	cudaMemcpy(h_energies, d_energies, bondCount*sizeof(float), cudaMemcpyDeviceToHost);
	int i;
	float energy = 0.0f;
	for(i = 0; i < bondCount; i++){
		energy += h_energies[i];
	}
	return energy;
}
