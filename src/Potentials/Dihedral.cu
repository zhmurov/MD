/*
 * Dihedral.cu
 *
 *  Created on: 17.10.2017
 *      Author: zhmurov
 */

#include "Dihedral.cuh"

/*
 * Constructor for proper and improper dihedral potential
 * \param mdd is the pointer to the main MDData object
 * \param dihedralCount is a total number of dihedrals in the system
 * \param dihedrals is an int4 array of size dihedralCount, containing i-j-k-l indexes of dihedraled atoms as x,y, z and w of the int4 type
 * \param angleParameters is a float3 array of size dihedralCount, containing dihedral angle parameters: the equilibrium dihedral phi0 as x and Kphi as y and multiplicity as z fields of float3
 * If multiplicity is set to zero this angle considered improper
 *
 */
Dihedral::Dihedral(MDData *mdd, int dihedralCount, int4* dihedrals, float3* dihedralParameters){
	printf("Initializing proper dihedral potential\n");
	this->mdd = mdd;
	
	this->dihedralCount = dihedralCount;
	this->lastStepEnergyComputed = 0;

	lastDihedraled = 0;
	int d, i, j, k, l;
	for(d = 0; d < dihedralCount; d++){
		i = dihedrals[d].x;
		j = dihedrals[d].y;
		k = dihedrals[d].z;
		l = dihedrals[d].w;
		if(i > lastDihedraled){
			lastDihedraled = i;
		}
		if(j > lastDihedraled){
			lastDihedraled = j;
		}
		if(k > lastDihedraled){
			lastDihedraled = k;
		}
		if(l > lastDihedraled){
			lastDihedraled = l;
		}
	}

	h_dihedrals = (int4*)calloc(dihedralCount, sizeof(int4));
	h_pars = (float3*)calloc(dihedralCount, sizeof(float3));
	h_refs = (int4*)calloc(dihedralCount, sizeof(int4));
	h_count = (int*)calloc(lastDihedraled + 1, sizeof(int));

	cudaMalloc((void**)&d_dihedrals, dihedralCount*sizeof(int4));
	cudaMalloc((void**)&d_pars, dihedralCount*sizeof(float3));
	cudaMalloc((void**)&d_refs, dihedralCount*sizeof(int4));
	cudaMalloc((void**)&d_count, (lastDihedraled + 1)*sizeof(int));

	for(d = 0; d < dihedralCount; d++){
		i = dihedrals[d].x;
		j = dihedrals[d].y;
		k = dihedrals[d].z;
		l = dihedrals[d].w;
		h_dihedrals[d].x = i;
		h_dihedrals[d].y = j;
		h_dihedrals[d].z = k;
		h_dihedrals[d].w = l;
		h_pars[d].x = dihedralParameters[d].x;
		h_pars[d].y = dihedralParameters[d].y;
		h_pars[d].z = dihedralParameters[d].z;
		h_refs[d].x = h_count[i];
		h_refs[d].y = h_count[j];
		h_refs[d].z = h_count[k];
		h_refs[d].w = h_count[l];
		h_count[i]++;
		h_count[j]++;
		h_count[k]++;
		h_count[l]++;
	}

	blockCount = (dihedralCount - 1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int maxDihedrals = 0;
	for(i = 0; i <= lastDihedraled; i++){
		if(h_count[i] > maxDihedrals){
			maxDihedrals = h_count[i];
		}
	}

	widthTot = ((lastDihedraled)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;

	h_forces = (float4*)calloc(maxDihedrals*widthTot, sizeof(float4));
	cudaMalloc((void**)&d_forces, maxDihedrals*widthTot*sizeof(float4));

	blockCountSum = lastDihedraled/DEFAULT_BLOCK_SIZE + 1;
	blockSizeSum = DEFAULT_BLOCK_SIZE;

	h_energies = (float*)calloc(dihedralCount, sizeof(float));
	cudaMalloc((void**)&d_energies, dihedralCount*sizeof(float));

	cudaMemcpy(d_dihedrals, h_dihedrals, dihedralCount*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pars, h_pars, dihedralCount*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_refs, h_refs, dihedralCount*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_count, h_count, (lastDihedraled + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_energies, h_energies, dihedralCount*sizeof(float), cudaMemcpyHostToDevice);

	printf("Done initializing proper dihedral potential\n");
}

Dihedral::~Dihedral(){
	free(h_dihedrals);
	free(h_refs);
	free(h_pars);
	free(h_count);
	free(h_forces);
	free(h_energies);
	cudaFree(d_dihedrals);
	cudaFree(d_refs);
	cudaFree(d_pars);
	cudaFree(d_count);
	cudaFree(d_forces);
	cudaFree(d_energies);
}

__global__ void dihedralCompute_kernel(int4* d_dihedrals, int4* d_refs, float3* d_pars, float4* d_forces, int widthTot, int dihedralCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < dihedralCount){

		int4 dihedral = d_dihedrals[d_i];
		int4 ref = d_refs[d_i];
		float3 pars = d_pars[d_i];

		float4 r1 = c_mdd.d_coord[dihedral.x];
		float4 r2 = c_mdd.d_coord[dihedral.y];
		float4 r3 = c_mdd.d_coord[dihedral.z];
		float4 r4 = c_mdd.d_coord[dihedral.w];

		float3 dr12, dr23, dr34;

		float3 pb = c_mdd.bc.len;

		dr12.x = r1.x - r2.x;
		dr12.y = r1.y - r2.y;
		dr12.z = r1.z - r2.z;

		dr12.x -= rint(dr12.x/pb.x)*pb.x;
		dr12.y -= rint(dr12.y/pb.y)*pb.y;
		dr12.z -= rint(dr12.z/pb.z)*pb.z;

		dr23.x = r2.x - r3.x;
		dr23.y = r2.y - r3.y;
		dr23.z = r2.z - r3.z;

		dr23.x -= rint(dr23.x/pb.x)*pb.x;
		dr23.y -= rint(dr23.y/pb.y)*pb.y;
		dr23.z -= rint(dr23.z/pb.z)*pb.z;

		dr34.x = r3.x - r4.x;
		dr34.y = r3.y - r4.y;
		dr34.z = r3.z - r4.z;

		dr34.x -= rint(dr34.x/pb.x)*pb.x;
		dr34.y -= rint(dr34.y/pb.y)*pb.y;
		dr34.z -= rint(dr34.z/pb.z)*pb.z;

		//float r232 = dr23.x*dr23.x + dr23.y*dr23.y + dr23.z*dr23.z;

		float4 a, b, c;

		a.x = dr12.y*dr23.z - dr12.z*dr23.y;
		a.y = dr12.z*dr23.x - dr12.x*dr23.z;
		a.z = dr12.x*dr23.y - dr12.y*dr23.x;
		a.w = 1.0f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);

		b.x = dr23.y*dr34.z - dr23.z*dr34.y;
		b.y = dr23.z*dr34.x - dr23.x*dr34.z;
		b.z = dr23.x*dr34.y - dr23.y*dr34.x;
		b.w = 1.0f/sqrtf(b.x*b.x + b.y*b.y + b.z*b.z);

		c.x = dr23.y*a.z - dr23.z*a.y;
		c.y = dr23.z*a.x - dr23.x*a.z;
		c.z = dr23.x*a.y - dr23.y*a.x;
		c.w = 1.0f/sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);

		float coschi = (a.x*b.x + a.y*b.y + a.z*b.z)*a.w*b.w;
		//float sinchi = -sqrtf(r232)*a.w*b.w*(a.x*dr34.x + a.y*dr34.y + a.z*dr34.z);//(c.x*b.x + c.y*b.y + c.z*b.z)*c.w*b.w;
		float sinchi = (c.x*b.x + c.y*b.y + c.z*b.z)*c.w*b.w;

		float chi = -atan2(sinchi, coschi);

		float mult = 0.0f;

		if(pars.z > 0.0f){
			mult = -pars.z*pars.y*sinf(pars.z*chi - pars.x);
		/*
		//  CHARMM Implementation:
			int i;
			int n = (int)pars.z;
			float E1 = 1.0f;
			float df = 0.0f;
			float ddf;

			for(i = 0; i < n; i++){
				ddf = E1*coschi - df*sinchi;
				df = E1*sinchi + df*coschi;
				E1 = ddf;
			}
			mult = pars.z*pars.y*(df*cosf(pars.x) - ddf*sinf(pars.x));//sinf(pars.x*chi - pars.x);*/
		} else {
			float diff = chi - pars.x;
			if(diff < -M_PI){
				diff += 2.0f*M_PI;
			} else
			if(diff > M_PI){
				diff -= 2.0f*M_PI;
			}
			mult = 2.0f*pars.y*diff;
		}

		/*
		// CHARMM implementation:
		float4 f1, f2, f3;

		f1.w = mult*a.w*a.w*sqrtf(r232);
		f1.x = f1.w*a.x;
		f1.y = f1.w*a.y;
		f1.z = f1.w*a.z;

		f2.w = -mult/sqrtf(r232);
		float r1223 = dr12.x*dr23.x + dr12.y*dr23.y + dr12.z*dr23.z;
		float r2334 = dr23.x*dr34.x + dr23.y*dr34.y + dr23.z*dr34.z;
		float m1 = f2.w*r1223*a.w*a.w;
		float m2 = f2.w*r2334*b.w*b.w;
		f2.x = m1*a.x + m2*b.x;
		f2.y = m1*a.y + m2*b.y;
		f2.z = m1*a.z + m2*b.z;

		f3.w = mult*b.w*b.w*sqrtf(r232);
		f3.x = f3.w*b.x;
		f3.y = f3.w*b.y;
		f3.z = f3.w*b.z;*/

		float4 f1, f2, f3;

		b.x *= b.w;
		b.y *= b.w;
		b.z *= b.w;

		if(fabs(sinchi) > 0.1f){

			a.x *= a.w;
			a.y *= a.w;
			a.z *= a.w;

			float3 dcosda, dcosdb;

			dcosda.x = a.w*(coschi*a.x - b.x);
			dcosda.y = a.w*(coschi*a.y - b.y);
			dcosda.z = a.w*(coschi*a.z - b.z);

			dcosdb.x = b.w*(coschi*b.x - a.x);
			dcosdb.y = b.w*(coschi*b.y - a.y);
			dcosdb.z = b.w*(coschi*b.z - a.z);

			mult = mult/sinchi;

			f1.x = mult*(dr23.y*dcosda.z - dr23.z*dcosda.y);
			f1.y = mult*(dr23.z*dcosda.x - dr23.x*dcosda.z);
			f1.z = mult*(dr23.x*dcosda.y - dr23.y*dcosda.x);

			f3.x = mult*(dr23.z*dcosdb.y - dr23.y*dcosdb.z);
			f3.y = mult*(dr23.x*dcosdb.z - dr23.z*dcosdb.x);
			f3.z = mult*(dr23.y*dcosdb.x - dr23.x*dcosdb.y);

			f2.x = mult*(dr12.z*dcosda.y - dr12.y*dcosda.z + dr34.y*dcosdb.z - dr34.z*dcosdb.y);
			f2.y = mult*(dr12.x*dcosda.z - dr12.z*dcosda.x + dr34.z*dcosdb.x - dr34.x*dcosdb.z);
			f2.z = mult*(dr12.y*dcosda.x - dr12.x*dcosda.y + dr34.x*dcosdb.y - dr34.y*dcosdb.x);

		} else {

			c.x *= c.w;
			c.y *= c.w;
			c.z *= c.w;

			float3 dsindc, dsindb;

			dsindc.x = c.w*(sinchi*c.x - b.x);
			dsindc.y = c.w*(sinchi*c.y - b.y);
			dsindc.z = c.w*(sinchi*c.z - b.z);

			dsindb.x = b.w*(sinchi*b.x - c.x);
			dsindb.y = b.w*(sinchi*b.y - c.y);
			dsindb.z = b.w*(sinchi*b.z - c.z);

			mult = -mult/coschi;

			f1.x = mult*((dr23.y*dr23.y + dr23.z*dr23.z)*dsindc.x - dr23.x*dr23.y*dsindc.y - dr23.x*dr23.z*dsindc.z);
			f1.y = mult*((dr23.z*dr23.z + dr23.x*dr23.x)*dsindc.y - dr23.y*dr23.z*dsindc.z - dr23.y*dr23.x*dsindc.x);
			f1.z = mult*((dr23.x*dr23.x + dr23.y*dr23.y)*dsindc.z - dr23.z*dr23.x*dsindc.x - dr23.z*dr23.y*dsindc.y);

			f3.x = mult*(dsindb.y*dr23.z - dsindb.z*dr23.y);
			f3.y = mult*(dsindb.z*dr23.x - dsindb.x*dr23.z);
			f3.z = mult*(dsindb.x*dr23.y - dsindb.y*dr23.x);

			f2.x = mult*(-(dr23.y*dr12.y + dr23.z*dr12.z)*dsindc.x + (2.0f*dr23.x*dr12.y - dr12.x*dr23.y)*dsindc.y
					+ (2.0f*dr23.x*dr12.z - dr12.x*dr23.z)*dsindc.z + dsindb.z*dr34.y - dsindb.y*dr34.z);
			f2.y = mult*(-(dr23.z*dr12.z + dr23.x*dr12.x)*dsindc.y + (2.0f*dr23.y*dr12.z - dr12.y*dr23.z)*dsindc.z
					+ (2.0f*dr23.y*dr12.x - dr12.y*dr23.x)*dsindc.x + dsindb.x*dr34.z - dsindb.z*dr34.x);
			f2.z = mult*(-(dr23.x*dr12.x + dr23.y*dr12.y)*dsindc.z + (2.0f*dr23.z*dr12.x - dr12.z*dr23.x)*dsindc.x
					+ (2.0f*dr23.z*dr12.y - dr12.z*dr23.y)*dsindc.y + dsindb.y*dr34.x - dsindb.x*dr34.y);
		}

		d_forces[widthTot*ref.x + dihedral.x] = f1;
		f1.x = f2.x - f1.x;
		f1.y = f2.y - f1.y;
		f1.z = f2.z - f1.z;
		d_forces[widthTot*ref.y + dihedral.y] = f1;
		f2.x = f3.x - f2.x;
		f2.y = f3.y - f2.y;
		f2.z = f3.z - f2.z;
		d_forces[widthTot*ref.z + dihedral.z] = f2;
		f3.x = -f3.x;
		f3.y = -f3.y;
		f3.z = -f3.z;
		d_forces[widthTot*ref.w + dihedral.w] = f3;
	}
}

__global__ void dihedralSum_kernel(int* d_count, float4* d_forces, int widthTot, int lastDihedraled){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i <= lastDihedraled){
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

void Dihedral::compute(){
	dihedralCompute_kernel<<<this->blockCount, this->blockSize>>>(d_dihedrals, d_refs, d_pars, d_forces, widthTot, dihedralCount);
	//cudaThreadSynchronize();
	dihedralSum_kernel<<<blockCountSum, blockSizeSum>>>(d_count, d_forces, widthTot, lastDihedraled);
}


__global__ void dihedralEnergy_kernel(int4* d_dihedrals, float3* d_pars, float* d_energies, int dihedralCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < dihedralCount){

		int4 dihedral = d_dihedrals[d_i];
		float3 pars = d_pars[d_i];

		float4 r1 = c_mdd.d_coord[dihedral.x];
		float4 r2 = c_mdd.d_coord[dihedral.y];
		float4 r3 = c_mdd.d_coord[dihedral.z];
		float4 r4 = c_mdd.d_coord[dihedral.w];

		float3 dr12, dr23, dr34;
		float3 pb = c_mdd.bc.len;

		dr12.x = r1.x - r2.x;
		dr12.y = r1.y - r2.y;
		dr12.z = r1.z - r2.z;

		dr12.x -= rint(dr12.x/pb.x)*pb.x;
		dr12.y -= rint(dr12.y/pb.y)*pb.y;
		dr12.z -= rint(dr12.z/pb.z)*pb.z;

		dr23.x = r2.x - r3.x;
		dr23.y = r2.y - r3.y;
		dr23.z = r2.z - r3.z;

		dr23.x -= rint(dr23.x/pb.x)*pb.x;
		dr23.y -= rint(dr23.y/pb.y)*pb.y;
		dr23.z -= rint(dr23.z/pb.z)*pb.z;

		dr34.x = r3.x - r4.x;
		dr34.y = r3.y - r4.y;
		dr34.z = r3.z - r4.z;

		dr34.x -= rint(dr34.x/pb.x)*pb.x;
		dr34.y -= rint(dr34.y/pb.y)*pb.y;
		dr34.z -= rint(dr34.z/pb.z)*pb.z;

		float4 a, b, c;

		a.x = dr12.y*dr23.z - dr12.z*dr23.y;
		a.y = dr12.z*dr23.x - dr12.x*dr23.z;
		a.z = dr12.x*dr23.y - dr12.y*dr23.x;
		a.w = 1.0f/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);

		b.x = dr23.y*dr34.z - dr23.z*dr34.y;
		b.y = dr23.z*dr34.x - dr23.x*dr34.z;
		b.z = dr23.x*dr34.y - dr23.y*dr34.x;
		b.w = 1.0f/sqrtf(b.x*b.x + b.y*b.y + b.z*b.z);

		c.x = dr23.y*a.z - dr23.z*a.y;
		c.y = dr23.z*a.x - dr23.x*a.z;
		c.z = dr23.x*a.y - dr23.y*a.x;
		c.w = 1.0f/sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);

		float coschi = (a.x*b.x + a.y*b.y + a.z*b.z)*a.w*b.w;
		float sinchi = (c.x*b.x + c.y*b.y + c.z*b.z)*c.w*b.w;

		float chi = -atan2(sinchi, coschi);

		float pot;
		if(pars.z > 0.0f){
			pot = pars.y*(1.0f + cosf(pars.z*chi - pars.x));
		} else {
			float diff = chi - pars.x;
			if(diff < -M_PI){
				diff += 2.0f*M_PI;
			} else
			if(diff > M_PI){
				diff -= 2.0f*M_PI;
			}
			pot = pars.y*diff*diff;
		}
		d_energies[d_i] = pot;
	}
}


float Dihedral::getEnergies(int energyId, int timestep){
	if(timestep != lastStepEnergyComputed){
		dihedralEnergy_kernel<<<this->blockCount, this->blockSize>>>(d_dihedrals, d_pars, d_energies, dihedralCount);
		//cudaThreadSynchronize();
		cudaMemcpy(h_energies, d_energies, dihedralCount*sizeof(float), cudaMemcpyDeviceToHost);
		int i;
		energyProper = 0.0f;
		energyImproper = 0.0f;
		for(i = 0; i < dihedralCount; i++){
			if(h_pars[i].z > 0){
				energyProper += h_energies[i];
			} else {
				energyImproper += h_energies[i];
			}
		}
		lastStepEnergyComputed = timestep;
	}
	if(energyId == 0){
		return energyProper;
	} else {
		return energyImproper;
	}
}

