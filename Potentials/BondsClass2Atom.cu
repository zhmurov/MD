/*
 * BondsClass2Atom.cu
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#include "BondsClass2Atom.cuh"

/*
float4* bondCoeffs; //from paramFile
bondCoeffs[i].x = l0;
bondCoeffs[i].y = k2;
bondCoeffs[i].z = k3;
bondCoeffs[i].w = k4;

int4* pair; //from topFile
pair[i].x = i;
pair[i].y = j;
pair[i].z = func;
*/

BondsClass2Atom::BondsClass2Atom(MDData *mdd, int bondCountPar, int bondCountTop, int4* pair, float4* bondCoeffs){
	printf("Initializing BondsClass2Atom potential\n");
	this->mdd = mdd;
	int i;
	h_bd.bondPar = (float4*)calloc(bondCount, sizeof(float4));
	cudaMalloc((void**)&d_bd.bondPar, bondCount*sizeof(float4));
	for(i = 0; i < bondCount; i++){
		h_bd.bondPar[i].x = bondCoeffs[i].x;
		h_bd.bondPar[i].y = 2.0f*bondCoeffs[i].y;
		h_bd.bondPar[i].z = 3.0f*bondCoeffs[i].z;
		h_bd.bondPar[i].w = 4.0f*bondCoeffs[i].w;
	}

	cudaMemcpy(d_bd.bondPar, h_bd.bondPar, bondCount*sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTexture(0, t_bondPar, d_bd.bondPar, bondCount*sizeof(float4));

	// Count bonds per atom

	h_bd.bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bd.bondCount, mdd->N*sizeof(int));

	for (i=0; i < bondCount; i++) {
		h_bd.bondCount[pair[i].x]++;
		h_bd.bondCount[pair[i].y]++;
	}

	// Calculate maxBond and lastBonded

	int maxBonds = 0;
	int lastBonded = 0;
	for(i = 0; i < mdd->N; i++){
		if(h_bd.bondCount[i] > 0){
			lastBonded = i;
		}
		if(maxBonds < h_bd.bondCount[i]){
			maxBonds = h_bd.bondCount[i];
		}
	}

	bondedCount = lastBonded + 1;
	this->blockCount = (bondedCount-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	widthTot = ((bondedCount-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	h_bd.bonds = (int2*)calloc(widthTot*maxBonds, sizeof(int2));
	cudaMalloc((void**)&d_bd.bonds, widthTot*maxBonds*sizeof(int2));

	// Fill list of bonded neighbours for each atom

	for(i = 0; i < mdd->N; i++){
		h_bd.bondCount[i] = 0;
	}
	int a1, a2, type;
	for (i = 0; i < bondCount; i++) {
		a1 = pair[i].x;
		a2 = pair[i].y;
		type = pair[i].z;
		h_bd.bonds[h_bd.bondCount[a1]*widthTot + a1].x = a2;
		h_bd.bonds[h_bd.bondCount[a2]*widthTot + a2].x = a1;
		h_bd.bonds[h_bd.bondCount[a1]*widthTot + a1].y = type;
		h_bd.bonds[h_bd.bondCount[a2]*widthTot + a2].y = type;
		h_bd.bondCount[a1]++;
		h_bd.bondCount[a2]++;
	}

	h_bd.energies = (float*)calloc(bondedCount, sizeof(float));
	cudaMalloc((void**)&d_bd.energies, bondedCount*sizeof(float));

	cudaMemcpy(d_bd.bondCount, h_bd.bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bd.bonds, h_bd.bonds, widthTot*maxBonds*sizeof(int2), cudaMemcpyHostToDevice);
}




/*BondsClass2Atom::BondsClass2Atom(MDData *mdd, std::vector<int3> &bonds, std::vector<Coeffs> &parameters)
{
	this->mdd = mdd;

	int i;

	// Coefficents

	int btCount = parameters.size();
	h_bd.bondPar = (float4*)calloc(btCount, sizeof(float4));
	cudaMalloc((void**)&d_bd.bondPar, btCount*sizeof(float4));
	for (i=0;i!=parameters.size();i++) {
		Coeffs& coeffs = parameters[i];
		h_bd.bondPar[i].x = atof(coeffs.coeffs.at(0).c_str()); // l0
		h_bd.bondPar[i].y = 2.0f*atof(coeffs.coeffs.at(1).c_str()); // 2*k2
		h_bd.bondPar[i].z = 3.0f*atof(coeffs.coeffs.at(2).c_str()); // 3*k3
		h_bd.bondPar[i].w = 4.0f*atof(coeffs.coeffs.at(3).c_str()); // 4*k4
	}

	cudaMemcpy(d_bd.bondPar, h_bd.bondPar, btCount*sizeof(float4), cudaMemcpyHostToDevice);
	cudaBindTexture(0, t_bondPar, d_bd.bondPar, btCount*sizeof(float4));

	// Count bonds per atom

	h_bd.bondCount = (int*)calloc(mdd->N, sizeof(int));
	cudaMalloc((void**)&d_bd.bondCount, mdd->N*sizeof(int));
	for (i=0;i!=bonds.size();i++) {
		h_bd.bondCount[bonds[i].x]++;
		h_bd.bondCount[bonds[i].y]++;
	}

	// Calculate maxBond and lastBonded

	int maxBonds = 0;
	int lastBonded = 0;
	for(i = 0; i < mdd->N; i++){
		if(h_bd.bondCount[i] > 0){
			lastBonded = i;
		}
		if(maxBonds < h_bd.bondCount[i]){
			maxBonds = h_bd.bondCount[i];
		}
	}

	bondedCount = lastBonded + 1;
	this->blockCount = (bondedCount-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	widthTot = ((bondedCount-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	h_bd.bonds = (int2*)calloc(widthTot*maxBonds, sizeof(int2));
	cudaMalloc((void**)&d_bd.bonds, widthTot*maxBonds*sizeof(int2));

	// Fill list of bonded neighbours for each atom

	for(i = 0; i < mdd->N; i++){
		h_bd.bondCount[i] = 0;
	}
	int a1, a2, type;
	for (i=0;i!=bonds.size();i++) {
		a1 = bonds[i].x;
		a2 = bonds[i].y;
		type = bonds[i].z;
		h_bd.bonds[h_bd.bondCount[a1]*widthTot + a1].x = a2;
		h_bd.bonds[h_bd.bondCount[a2]*widthTot + a2].x = a1;
		h_bd.bonds[h_bd.bondCount[a1]*widthTot + a1].y = type;
		h_bd.bonds[h_bd.bondCount[a2]*widthTot + a2].y = type;
		h_bd.bondCount[a1]++;
		h_bd.bondCount[a2]++;
	}

	h_bd.energies = (float*)calloc(bondedCount, sizeof(float));
	cudaMalloc((void**)&d_bd.energies, bondedCount*sizeof(float));

	cudaMemcpy(d_bd.bondCount, h_bd.bondCount, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bd.bonds, h_bd.bonds, widthTot*maxBonds*sizeof(int2), cudaMemcpyHostToDevice);
}*/

BondsClass2Atom::~BondsClass2Atom(){
	free(h_bd.bondCount);
	free(h_bd.bonds);
	free(h_bd.bondPar);
	free(h_bd.energies);
	cudaFree(d_bd.bondCount);
	cudaFree(d_bd.bonds);
	cudaFree(d_bd.bondPar);
	cudaFree(d_bd.energies);
}

__global__ void bondClass2_kernel(int* d_bondCount, int2* d_bonds, int widthTot, int bondedCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondedCount){
		int j;
		float4 f = c_mdd.d_force[d_i];
		float4 r1 = tex1Dfetch(t_coord, d_i);
		float4 r2;
		for(j = 0; j < d_bondCount[d_i]; j++){
			int2 bond = d_bonds[j*widthTot + d_i];
			r2 = tex1Dfetch(t_coord, bond.x);
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float4 par = tex1Dfetch(t_bondPar, bond.y);
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

		/*	if (d_i==20) {
				printf("%d %d %f\n", d_i, bond.x, r2.z*df);
				printf("%f %f %f %f\n", par.x, par.y, par.z, par.w);
				printf("%f %f %f %f\n", r, r2.w, df, r2.z);
				float4 r3 = tex1Dfetch(t_coord, bond.x);
				printf("(%f %f %f) (%f %f %f)\n", r1.x, r1.y, r1.z, r3.x, r3.y, r3.z);
			}*/
		}
		c_mdd.d_force[d_i] = f;
	}
}


void BondsClass2Atom::compute(MDData *mdd){
	bondClass2_kernel<<<this->blockCount, this->blockSize>>>(d_bd.bondCount, d_bd.bonds, widthTot, bondedCount);
	/*int i;
	cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	FILE* file = fopen("bond_forces.dat", "w");
	for(i = 0; i < mdd->N; i++){
		fprintf(file, "%f %f %f\n", mdd->h_force[i].x, mdd->h_force[i].y, mdd->h_force[i].z);
	}
	fclose(file);
	exit(0);*/
}

__global__ void bondClass2Energy_kernel(int* d_bondCount, int2* d_bonds, float* d_energies, int widthTot, int bondedCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < bondedCount){
		int j;
		float4 r1 = tex1Dfetch(t_coord, d_i);
		float4 r2;
		float pot = 0.0f;
		for(j = 0; j < d_bondCount[d_i]; j++){
			int2 bond = d_bonds[j*widthTot + d_i];
			r2 = tex1Dfetch(t_coord, bond.x);
			r2.x -= r1.x;
			r2.y -= r1.y;
			r2.z -= r1.z;

			float3 pb = c_mdd.bc.len;
			r2.x -= rint(r2.x/pb.x)*pb.x;
			r2.y -= rint(r2.y/pb.y)*pb.y;
			r2.z -= rint(r2.z/pb.z)*pb.z;

			float4 par = tex1Dfetch(t_bondPar, bond.y);
			r2.w = r2.x*r2.x + r2.y*r2.y + r2.z*r2.z;
			r1.w = sqrtf(r2.w);
			r1.w -= par.x;
			r2.w = r1.w*r1.w;

			pot += (0.5f*par.y + (par.z/3.0f)*r1.w + 0.25f*par.w*r2.w)*r2.w;

		}
		d_energies[d_i] = pot;
	}
}

float BondsClass2Atom::get_energies(int energy_id, int timestep){
	bondClass2Energy_kernel<<<this->blockCount, this->blockSize>>>(d_bd.bondCount, d_bd.bonds, d_bd.energies, widthTot, bondedCount);
	//cudaThreadSynchronize();
	cudaMemcpy(h_bd.energies, d_bd.energies, bondedCount*sizeof(float), cudaMemcpyDeviceToHost);
	int i;
	float energy = 0.0f;
	for(i = 0; i < bondedCount; i++){
		energy += h_bd.energies[i];
	}
	energy *= 0.5f;
	return energy;
}
