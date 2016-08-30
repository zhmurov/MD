/*
 * md.cu
 *
 *  Created on: 15.08.2012
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */

#include "md.cuh"

// Util
#include "Util/ReductionAlgorithms.cu"
#include "Util/ReductionAlgorithmsFloat4.cu"

// Potentials
#include "Potentials/BondsClass2Atom.cu"
//#include "Potentials/BondsClass2Pair.cu"
#include "Potentials/AngleClass2.cu"
#include "Potentials/GaussExcluded.cu"
#include "Potentials/Langevin.cu"
#include "Potentials/PPPM.cu"
#include "Potentials/Coulomb.cu"

// Updaters
#include "Updaters/CoordinatesOutputDCD.cu"
#include "Updaters/EnergyOutput.cu"
#include "Updaters/PairlistUpdater.cu"
#include "Updaters/PairListL1.cu"
#include "Updaters/PairListL2.cu"
#include "Updaters/FixMomentum.cu"

// Integrators
#include "Integrators/LeapFrog.cu"
#include "Integrators/VelocityVerlet.cu"
#include "Integrators/LeapFrogNoseHoover.cu"

void dumpPSF(char* filename, TOPData &top){
	printf("Creating psf...\n");
	PSF psf;
	psf.natom = top.atomCount;
	psf.ntheta = 0;
	psf.nphi = 0;
	psf.nimphi = 0;
	psf.nnb = 0;
	psf.ncmap = 0;
	psf.atoms = (PSFAtom*)calloc(psf.natom, sizeof(PSFAtom));
	int i;

	for(i = 0; i < top.atomCount; i++){
		psf.atoms[i].id = top.atoms[i].id;
		psf.atoms[i].m = top.atoms[i].mass;

		sprintf(psf.atoms[i].name, "C");
		sprintf(psf.atoms[i].type, "%s", top.atoms[i].type);
		psf.atoms[i].q = top.atoms[i].charge;
		if(top.atoms[i].resid == 0){
			sprintf(psf.atoms[i].resName, "ION");
		} else {
			sprintf(psf.atoms[i].resName, "DNA");
		}
		psf.atoms[i].resid = top.atoms[i].resid;
		sprintf(psf.atoms[i].segment, "%d", top.atoms[i].id);
	}

	psf.nbond = 0;
	for(i = 0; i < top.bondCount; i++){
		if(top.bonds[i].c0 == 1){
			psf.nbond ++;
		}
	}
	psf.bonds = (PSFBond*)calloc(psf.nbond, sizeof(PSFBond));
	int currentBond = 0;
	for(i = 0; i < top.bondCount; i++){
		if(top.bonds[i].c0 == 1){
			psf.bonds[currentBond].i = top.bonds[i].i;
			psf.bonds[currentBond].j = top.bonds[i].j;
			currentBond++;
		}
	}

	writePSF(filename, &psf);
	free(psf.atoms);
	free(psf.bonds);
}

void readCoordinatesFromFile(char* filename, MDData mdd){
	XYZ xyz;
	readXYZ(filename, &xyz);
	int i;
	for(i = 0; i < xyz.atomCount; i++){
		mdd.h_coord[i].x = xyz.atoms[i].x;
		mdd.h_coord[i].y = xyz.atoms[i].y;
		mdd.h_coord[i].z = xyz.atoms[i].z;
	}
}



void MDGPU::init()
{
	TOPData top;
	PARAMData par;

	char filename[FILENAME_LENGTH];
	getMaskedParameter(filename, PARAMETER_TOPOLOGY_FILENAME);
	readTOP(filename, &top);

	getMaskedParameter(filename, PARAMETER_PARAMETERS_FILENAME);
	readPARAM(filename, &par);

	getMaskedParameter(filename, PARAMETER_PSF_OUTPUT_FILENAME);
	dumpPSF(filename, top);

	cudaSetDevice(getIntegerParameter(PARAMETER_GPU_DEVICE));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	mdd.N = top.atomCount;
	printf("mdd.N\t%d\n", mdd.N);
	mdd.widthTot = ((mdd.N-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	mdd.dt = getFloatParameter(PARAMETER_TIMESTEP);
	mdd.numsteps = getIntegerParameter(PARAMETER_NUMSTEPS);

	mdd.ftm2v = FTM2V;

	mdd.bc.rlo.x = getFloatParameter("pbc_xlo", 0, 0);
	mdd.bc.rlo.y = getFloatParameter("pbc_ylo", 0, 0);
	mdd.bc.rlo.z = getFloatParameter("pbc_zlo", 0, 0);

	mdd.bc.rhi.x = getFloatParameter("pbc_xhi", 0, 0);
	mdd.bc.rhi.y = getFloatParameter("pbc_yhi", 0, 0);
	mdd.bc.rhi.z = getFloatParameter("pbc_zhi", 0, 0);

	mdd.bc.len.x = mdd.bc.rhi.x - mdd.bc.rlo.x;
	mdd.bc.len.y = mdd.bc.rhi.y - mdd.bc.rlo.y;
	mdd.bc.len.z = mdd.bc.rhi.z - mdd.bc.rlo.z;

	mdd.h_coord = (float4*)calloc(mdd.N, sizeof(float4));
	mdd.h_vel = (float4*)calloc(mdd.N, sizeof(float4));
	mdd.h_force = (float4*)calloc(mdd.N, sizeof(float4));
	mdd.h_mass = (float*)calloc(mdd.N, sizeof(float));
	mdd.h_charge = (float*)calloc(mdd.N, sizeof(float));
	mdd.h_atomTypes = (int*)calloc(mdd.N, sizeof(int));
	mdd.h_boxids = (int4*)calloc(mdd.N, sizeof(int4));

	getMaskedParameter(filename, PARAMETER_COORDINATES_FILENAME, "NONE");

	if(strncmp(filename, "NONE", 4) != 0){
		readCoordinatesFromFile(filename, mdd);
	}

	cudaMalloc((void**)&mdd.d_coord, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_vel, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_force, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_mass, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_charge, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_atomTypes, mdd.N*sizeof(int));
	cudaMalloc((void**)&mdd.d_boxids, mdd.N*sizeof(int4));


	int i, j;
	for(i = 0; i < mdd.N; i++){
		mdd.h_charge[i] = top.atoms[i].charge;
		mdd.h_atomTypes[i] = atoi(top.atoms[i].type) - 1;
	}

	for(i = 0; i < mdd.N; i++){
		for(j = 0; j < top.atomCount; j++){
			if(atoi(top.atoms[i].type) == atoi(top.atoms[j].type)){
				mdd.h_mass[i] = top.atoms[j].mass;
			}
		}
	}
	double totalMass = 0.0;
	for(i = 0; i < mdd.N; i++){
		totalMass += mdd.h_mass[i];
	}
	mdd.M = totalMass;

	int rseed = -getLongIntegerParameter(PARAMETER_RSEED);
	generateVelocities(getFloatParameter(PARAMETER_TEMPERATURE), &rseed);

	cudaMemcpy(mdd.d_coord, mdd.h_coord, mdd.N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_force, mdd.h_force, mdd.N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_vel, mdd.h_vel, mdd.N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_atomTypes, mdd.h_atomTypes, mdd.N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_mass, mdd.h_mass, mdd.N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_charge, mdd.h_charge, mdd.N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mdd.d_boxids, mdd.h_boxids, mdd.N*sizeof(int4), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_mdd, &mdd, sizeof(MDData), 0, cudaMemcpyHostToDevice);
	cudaBindTexture(0, t_coord, mdd.d_coord, mdd.N*sizeof(float4));
	cudaBindTexture(0, t_charges, mdd.d_charge, mdd.N*sizeof(float));
	cudaBindTexture(0, t_atomTypes, mdd.d_atomTypes, mdd.N*sizeof(int));

	// Add potentials, updaters and integrators

	char integ_str[PARAMETER_MAX_LENGTH];
	getMaskedParameter(integ_str, PARAMETER_INTEGRATOR);
	if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG) == 0) {
		integrator = new LeapFrog(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_VELOCITY_VERLET) == 0) {
		integrator = new VelocityVerlet(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER) == 0) {
		float tau = getFloatParameter(PARAMETER_NOSE_HOOVER_TAU);
		float T0 = getFloatParameter(PARAMETER_NOSE_HOOVER_T0);
		integrator = new LeapFrogNoseHoover(&mdd, tau, T0);
	} else {
		DIE("Integrator was set incorrectly!");
	}

	if(getYesNoParameter(PARAMETER_LANGEVIN, DEFAULT_LANGEVIN)){
		float damping = getFloatParameter(PARAMETER_DAMPING);
		int seed = getIntegerParameter(PARAMETER_LANGEVIN_SEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		potentials.push_back(new Langevin(&mdd, damping, seed, temperature));
	}
	
	//BondsClass2Atom
	//add top.func = 10
	int bondCountsPar = par.bondCount;
	int bondCountsTop = top.bondCount;

	int4* pair;
	pair = (int4*)calloc(bondCountsTop, sizeof(int4));

	for(int i = 0; i < bondCountsTop; i++){
		pair[i].x = top.bonds[i].i - 1;
		pair[i].y = top.bonds[i].j - 1;
		pair[i].z = top.bonds[i].c0 - 1;
	}

	float4* bondCoeffs;
	bondCoeffs = (float4*)calloc(bondCountsPar, sizeof(float4));

	for(int i = 0; i < bondCountsPar; i++){
		bondCoeffs[i].x = par.bondCoeff[i].l0;
		bondCoeffs[i].y = par.bondCoeff[i].k2;
		bondCoeffs[i].z = par.bondCoeff[i].k3;
		bondCoeffs[i].w = par.bondCoeff[i].k4;
	}

	potentials.push_back(new BondsClass2Atom(&mdd, bondCountsPar, bondCountsTop, pair, bondCoeffs));

	//AngleClass2
	//add top.func = 10
	int angleCountsPar = par.angleCount;
	int angleCountsTop = top.angleCount;

	int4* angle;
	angle = (int4*)calloc(angleCountsTop, sizeof(int4));

	for(int i = 0; i < angleCountsTop; i++){
		angle[i].x = top.angles[i].i - 1;
		angle[i].y = top.angles[i].j - 1;
		angle[i].z = top.angles[i].k - 1;
		angle[i].w = top.angles[i].c0 - 1;
	}

	float4* angleCoeffs;
	angleCoeffs = (float4*)calloc(angleCountsPar, sizeof(float4));

	for(int i = 0; i < angleCountsPar; i++){
		angleCoeffs[i].x = par.angleCoeff[i].theta0;
		angleCoeffs[i].y = par.angleCoeff[i].k2;
		angleCoeffs[i].z = par.angleCoeff[i].k3;
		angleCoeffs[i].w = par.angleCoeff[i].k4;
	}

	potentials.push_back(new AngleClass2(&mdd, angleCountsPar, angleCountsTop, angle, angleCoeffs));

	//Init pair lists

	float gausExclCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
	float coulCutoff = getFloatParameter(PARAMETER_COULOMB_CUTOFF);
	float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);

	int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
	int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

	std::vector<int2> exclusions(top.exclusionCount);
	for(int i = 0; i < top.exclusionCount; i++){
		if(top.exclusions[i].i < top.exclusions[i].j){
			exclusions[i].x = top.exclusions[i].i - 1;
			exclusions[i].y = top.exclusions[i].j - 1;
		}else{
			exclusions[i].x = top.exclusions[i].j - 1;
			exclusions[i].y = top.exclusions[i].i - 1;
		}
	}
	std::sort(exclusions.begin(), exclusions.end(), int2_comparatorEx);

	PairListL1* plistL1 = new PairListL1(&mdd, exclusions, possiblePairsCutoff, pairsCutoff, possiblePairsFreq);
	PairListL2* plistL2 = new PairListL2(&mdd, plistL1->d_pairs, pairsCutoff, coulCutoff, pairsFreq);
	updaters.push_back(plistL1);
	updaters.push_back(plistL2);

	//Gauss
	int typeCount = 1;
	bool boo;
	for (int i = 1; i < top.atomCount; i++){
		for(int j = 0; j < i; j++){
			if (atoi(top.atoms[j].type) == atoi(top.atoms[i].type)){ 
				boo = false;
				break;
			}else{
				boo = true;
			}
		}
		if (boo) {
			typeCount++;
		}
	}
	printf("typeCount = %d\n", typeCount);
	
	GaussExCoeff* gaussExCoeff;
	gaussExCoeff = (GaussExCoeff*)calloc(typeCount*typeCount, sizeof(GaussExCoeff));

	for(int i = 0; i < typeCount; i++){
		for(int j = 0; j < typeCount; j++){
			for(int k = 0; k < par.ljCount; k++){
				if((i == par.lj_RepulsiveCoeff[k].i - 1 && j == par.lj_RepulsiveCoeff[k].j - 1) || (j == par.lj_RepulsiveCoeff[k].i - 1 && i == par.lj_RepulsiveCoeff[k].j - 1)){
					gaussExCoeff[i+j*typeCount].A = par.lj_RepulsiveCoeff[k].A;
					gaussExCoeff[i+j*typeCount].l = par.lj_RepulsiveCoeff[k].l;
				}
			}
			for(int k = 0; k < par.gaussCount; k++){
				if((i == par.gaussCoeff[k].i - 1 && j == par.gaussCoeff[k].j - 1) || (j == par.gaussCoeff[k].i - 1 && i == par.gaussCoeff[k].j - 1)){
					gaussExCoeff[i+j*typeCount].numberGaussians = par.gaussCoeff[k].numberGaussians;
					gaussExCoeff[i+j*typeCount].B = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					gaussExCoeff[i+j*typeCount].C = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					gaussExCoeff[i+j*typeCount].R = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					for(int l = 0; l < par.gaussCoeff[k].numberGaussians; l++){
						gaussExCoeff[i+j*typeCount].B[l] = par.gaussCoeff[k].B[l];
						gaussExCoeff[i+j*typeCount].C[l] = par.gaussCoeff[k].C[l];
						gaussExCoeff[i+j*typeCount].R[l] = par.gaussCoeff[k].R[l];
					}
				}		
			}
		}
	}
	
	if(coulCutoff - gausExclCutoff > 10.0f){
		PairListL2* plistGausExcl = new PairListL2(&mdd, plistL2->d_pairs, coulCutoff, gausExclCutoff, pairsFreq);
		float cutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
		potentials.push_back(new GaussExcluded(&mdd, cutoff, typeCount, gaussExCoeff, plistGausExcl));
		updaters.push_back(plistGausExcl);
	} else {
		float cutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
		potentials.push_back(new GaussExcluded(&mdd, cutoff, typeCount, gaussExCoeff, plistL2));
	}

	float dielectric = getFloatParameter(PARAMETER_DIELECTRIC, DEFAULT_DIELECTRIC);
	PPPM* pppm = new PPPM(&mdd, dielectric, coulCutoff);
	potentials.push_back(pppm);
	potentials.push_back(new Coulomb(&mdd, plistL2, pppm->get_alpha(), dielectric, coulCutoff));

	updaters.push_back(new CoordinatesOutputDCD(&mdd));
	updaters.push_back(new EnergyOutput(&mdd, &potentials));
	
	if(getYesNoParameter(PARAMETER_FIX_MOMENTUM, DEFAULT_FIX_MOMENTUM)){
		updaters.push_back(new FixMomentum(&mdd, getIntegerParameter(PARAMETER_FIX_MOMENTUM_FREQUENCE)));
	}
}

void MDGPU::generateVelocities(float T, int * rseed){
	printf("Generating velocities at temperature T=%fK.\n", T);
	int i;
	if(T < 0){
		DIE("Negative value for temperature is set (T = %fK).", T);
	} else
	if(T == 0){
		for(i = 0; i < mdd.N; i++){
			mdd.h_vel[i].x = 0.0;
			mdd.h_vel[i].y = 0.0;
			mdd.h_vel[i].z = 0.0;
		}
	} else {
		for(i = 0; i < mdd.N; i++){
			double var = sqrt(BOLTZMANN_CONSTANT*T/mdd.h_mass[i]);
			mdd.h_vel[i].x = var*gasdev(rseed);
			mdd.h_vel[i].y = var*gasdev(rseed);
			mdd.h_vel[i].z = var*gasdev(rseed);
		}
	}
	float Temp = 0.0f;
	float Vav = 0.0f;
	int freq = getIntegerParameter(PARAMETER_ENERGY_OUTPUT_FREQUENCY);
	FILE* file = fopen("vels.dat", "w");
	for(i = 0; i < mdd.N; i++){
		mdd.h_vel[i].w = mdd.h_vel[i].x*mdd.h_vel[i].x + mdd.h_vel[i].y*mdd.h_vel[i].y + mdd.h_vel[i].z*mdd.h_vel[i].z;
		Vav += sqrtf(mdd.h_vel[i].w);
		fprintf(file, "%f\n", sqrtf(mdd.h_vel[i].w));
		Temp += mdd.h_vel[i].w*mdd.h_mass[i];;
		mdd.h_vel[i].w *= freq;
	}
	fclose(file);
	Temp /= mdd.N;
	Temp /= 3.0*BOLTZMANN_CONSTANT;
	Vav /= mdd.N;
	printf("Temperature of the system: %f (average velocity %f)\n", Temp, Vav);
}


void checkCUDAError(const char* msg) {
	//cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error == cudaSuccess)
		error = cudaThreadSynchronize();
	if (error == cudaSuccess)
		error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CudaError: %s: %s\n", msg, cudaGetErrorString(error));
exit(0);
	}
}

void MDGPU::compute()
{
	mdd.step = 0;
	int numsteps = mdd.numsteps;
	int nav = numsteps;
	int i;
	int u, p;
	for(u = 0; u != updaters.size(); u++){
		if(nav > updaters[u]->getFrequence()){
			nav = updaters[u]->getFrequence();
			cudaThreadSynchronize();
		}
	}

	cudaThreadSynchronize();
	for(p = 0; p != potentials.size(); p++){
		potentials[p]->compute(&mdd);
//checkCUDAError("Here!");
//printf("bla!");
		cudaThreadSynchronize();
	}

	/*cudaMemcpy(mdd.h_force, mdd.d_force, mdd.N*sizeof(float4), cudaMemcpyDeviceToHost);
	FILE* file = fopen("forces.dat", "w");
	for(i = 0; i < mdd.N; i++){
		fprintf(file, "%f %f %f\n", mdd.h_force[i].x, mdd.h_force[i].y, mdd.h_force[i].z);
	}
	fclose(file);
	file = fopen("coords.dat", "w");
	for(i = 0; i < mdd.N; i++){
		fprintf(file, "%f %f %f\n", mdd.h_coord[i].x, mdd.h_coord[i].y, mdd.h_coord[i].z);
	}
	fclose(file);
	exit(0);*/

	for(mdd.step = 0; mdd.step <= numsteps; mdd.step += nav){
		for(u = 0; u != updaters.size(); u++){
			if(mdd.step % updaters[u]->getFrequence() == 0){
				updaters[u]->update(&mdd);
				cudaThreadSynchronize();
			}
		}
		for(i = 0; i < nav; i++){
			integrator->integrate_step_one(&mdd);
			cudaThreadSynchronize();
			for(p = 0; p != potentials.size(); p++){
				potentials[p]->compute(&mdd);
				cudaThreadSynchronize();
				checkCUDAError("one of the potentials");
			}

			integrator->integrate_step_two(&mdd);
		}
	}

}

MDGPU::~MDGPU()
{
	free(mdd.h_coord);
	free(mdd.h_vel);
	free(mdd.h_force);
	free(mdd.h_mass);
	free(mdd.h_charge);
	free(mdd.h_atomTypes);

	cudaFree(mdd.d_coord);
	cudaFree(mdd.d_vel);
	cudaFree(mdd.d_force);
	cudaFree(mdd.d_mass);
	cudaFree(mdd.d_charge);
	cudaFree(mdd.d_atomTypes);
}

void compute(){

	MDGPU mdgpu;
	mdgpu.init();
	mdgpu.compute();
	cudaDeviceReset();
}

