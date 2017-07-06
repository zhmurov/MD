#include "Indentation.cuh"

Indentation::Indentation(MDData *mdd, int atomCount, int tipRadius, float3 tipCoord, float tipFriction, float3 baseCoord, int baseFreq, float3 baseDir, float baseVel, float ks, float eps, float sigm, float3 sfCoord, float3 sfN, float sfEps, float sfSigm, int dcdFreq, char* dcdCantFilename, char* indOutputFilename){
	this->mdd = mdd;
	this->atomCount = atomCount;
	this->tipRadius = tipRadius;
	this->tipCoord = tipCoord;
	this->tipFriction = tipFriction;
	this->baseCoord = baseCoord;
	this->baseFreq = baseFreq;
	this->baseDir = baseDir;
	this->baseVel = baseVel;
	this->ks = ks;
	this->eps = eps;
	this->sigm = sigm;
	this->sfCoord = sfCoord;
	this->sfN = sfN;
	this->sfEps = sfEps;
	this->sfSigm = sfSigm;
	this->dcdFreq = dcdFreq;

	const1 = -sfN.x*sfCoord.x - sfN.y*sfCoord.y - sfN.z*sfCoord.z;
	const2 = sqrt(sfN.x*sfN.x + sfN.y*sfN.y + sfN.z*sfN.z);

	tipDisplacement = 0.0f;
	baseDisplacement = 0.0f;

	tipCurrentCoord = tipCoord;

	sprintf(outputFilename, "%s", indOutputFilename);
	FILE* output = fopen(outputFilename, "w");
	fclose(output);

	// dcd
	int frameCount = mdd->numsteps/dcdFreq + 1;
	createDCD(&dcd_cant, 2, frameCount, 1, 0, dcdFreq, 1, mdd->bc.len.x, mdd->bc.len.y, mdd->bc.len.z);
	dcdOpenWrite(&dcd_cant, dcdCantFilename);
	dcdWriteHeader(dcd_cant);

	this->blockCount = (atomCount-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

	// force
	h_tipForce = (float3*)calloc(atomCount, sizeof(float3));
	cudaMalloc((void**)&d_tipForce, atomCount*sizeof(float3));
	cudaMemcpy(d_tipForce, h_tipForce, atomCount*sizeof(float3), cudaMemcpyHostToDevice);

	// energy
	h_energy = (float*)calloc(atomCount, sizeof(float));
	cudaMalloc((void**)&d_energy, atomCount*sizeof(float));
	cudaMemcpy(d_energy, h_energy, atomCount*sizeof(float), cudaMemcpyHostToDevice);
}

Indentation::~Indentation(){
	free(outputFilename);
	free(h_tipForce);
	free(h_energy);
	cudaFree(d_tipForce);
	cudaFree(d_energy);
}

__global__ void indentation_kernel(int atomCount, float tipRadius, float3 tipCurrentCoord, float eps, float sigm, float3 sfN, float sfEps, float sfSigm, float const1, float const2, float3* d_tipForce){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < atomCount){

		float rij_mod, df;
		float3 rij, rj;
		float4 ri, f;
		float temp;

		f = c_mdd.d_force[i];

		ri = c_mdd.d_coord[i];
		rj = tipCurrentCoord;

		rij.x = rj.x - ri.x;
		rij.y = rj.y - ri.y;
		rij.z = rj.z - ri.z;

		rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

	// repulsive potential
		temp = 6.0f*pow(sigm, 6.0f)/pow((rij_mod - tipRadius), 7.0f);
		df = -eps*temp/rij_mod;

		f.x += df*rij.x;
		f.y += df*rij.y;
		f.z += df*rij.z;

	// tip
		d_tipForce[i].x = -df*rij.x;
		d_tipForce[i].y = -df*rij.y;
		d_tipForce[i].z = -df*rij.z;

	// surface
		//rij_mod = abs(A*x + B*y + C*z + D)/sqrt(A^2 + B^2 + C^2)
		//const1 = D = -A*x0 - B*y0 - C*z0
		//const2 = sqrt(A^2 + B^2 + C^2)
		rij_mod = abs(sfN.x*ri.x + sfN.y*ri.y + sfN.z*ri.z + const1)/const2;

		temp = 6.0f*pow(sfSigm, 6.0f)/pow(rij_mod, 7.0f);
		df = sfEps*temp/rij_mod;

		//TODO abs ?
		f.x += df*rij_mod*sfN.x;
		f.y += df*rij_mod*sfN.y;
		f.z += df*rij_mod*sfN.z;

		c_mdd.d_force[i] = f;
	}
}

void Indentation::compute(){
	if(mdd->step % baseFreq == 0){
		baseDisplacement = baseVel*mdd->dt*mdd->step;
	}

	tipCurrentCoord.x = tipCoord.x + tipDisplacement*baseDir.x;
	tipCurrentCoord.y = tipCoord.y + tipDisplacement*baseDir.y;
	tipCurrentCoord.z = tipCoord.z + tipDisplacement*baseDir.z;

	indentation_kernel<<<this->blockCount, this->blockSize>>>(atomCount, tipRadius, tipCurrentCoord, eps, sigm, sfN, sfEps, sfSigm, const1, const2, d_tipForce);

	float mult = 0.0f;
	float3 resForce = make_float3(0.0f, 0.0f, 0.0f);
	cudaMemcpy(h_tipForce, d_tipForce, atomCount*sizeof(float3), cudaMemcpyDeviceToHost);
	for(int i = 0; i < atomCount; i++){
		resForce.x += h_tipForce[i].x;
		resForce.y += h_tipForce[i].y;
		resForce.z += h_tipForce[i].z;
	}
	mult = -ks*(tipDisplacement - baseDisplacement);

	resForce.x += baseDir.x*mult;
	resForce.y += baseDir.y*mult;
	resForce.z += baseDir.z*mult;

	// FRICTION COEFFICIENT
	// ksi = 6*pi*nu*r = 5.655E+4
	// 1/ksi = 1.77E-5 = 0.0000029
	tipDisplacement += tipFriction*mdd->dt*(resForce.x*baseDir.x + resForce.y*baseDir.y + resForce.z*baseDir.z);

	if (mdd->step % dcdFreq == 0){
		FILE* output = fopen(outputFilename, "a");
		fprintf(output, "%3.6f  ", (baseCoord.z + baseDisplacement*baseDir.z));
		fprintf(output, "%3.6f  ", (tipCoord.z + tipDisplacement*baseDir.z));
		fprintf(output, "%3.6f  ", baseDisplacement*baseDir.z);
		fprintf(output, "%3.6f  ", tipDisplacement*baseDir.z);
		fprintf(output, "%f  ", mult*baseDir.z);
		fprintf(output, "%f\n", (resForce.z - baseDir.z*mult));
		fclose(output);

		// base
		dcd_cant.frame.X[0] = (baseCoord.x + baseDisplacement*baseDir.x)*10.0f;		// [nm]->[angstr]
		dcd_cant.frame.Y[0] = (baseCoord.y + baseDisplacement*baseDir.y)*10.0f;		// [nm]->[angstr]
		dcd_cant.frame.Z[0] = (baseCoord.z + baseDisplacement*baseDir.z)*10.0f;		// [nm]->[angstr]
		// tip
		dcd_cant.frame.X[1] = (tipCoord.x + tipDisplacement*baseDir.x)*10.0f;		// [nm]->[angstr]
		dcd_cant.frame.Y[1] = (tipCoord.y + tipDisplacement*baseDir.y)*10.0f;		// [nm]->[angstr]
		dcd_cant.frame.Z[1] = (tipCoord.z + tipDisplacement*baseDir.z)*10.0f;		// [nm]->[angstr]
		dcdWriteFrame(dcd_cant);
	}
}

__global__ void indentationEnergy_kernel(int N, float tipRadius, float3 tipCurrentCoord, float eps, float sigm, float* d_energy){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < N){

		float rij_mod, energy;
		float3 rij, rj;
		float4 ri;

		ri = c_mdd.d_coord[i];
		rj = tipCurrentCoord;

		rij.x = rj.x - ri.x;
		rij.y = rj.y - ri.y;
		rij.z = rj.z - ri.z;

		rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

		energy = eps*pow(sigm, 6.0f)/pow((rij_mod - tipRadius), 6.0f);
		d_energy[i] = energy;
	}
}



float Indentation::getEnergies(int energyId, int Nstep){
	indentationEnergy_kernel<<<this->blockCount, this->blockSize>>>(atomCount, tipRadius, tipCurrentCoord, eps, sigm, d_energy);

	cudaMemcpy(h_energy, d_energy, atomCount*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < atomCount; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
