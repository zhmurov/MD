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
#include "Potentials/FENE.cu"
#include "Potentials/LJP.cu"
#include "Potentials/Repulsive.cu"
#include "Potentials/PushingSphere.cu"
#include "Potentials/Indentation.cu"

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

#include "Integrators/LeapFrog_new.cu"
#include "Integrators/LeapFrog_overdumped.cu"
#include "Integrators/SteepestDescent.cu"

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
		sprintf(psf.atoms[i].resName, "%s", top.atoms[i].resName);
		psf.atoms[i].resid = top.atoms[i].resid;
		sprintf(psf.atoms[i].segment, "%s", top.atoms[i].type);
	}

	psf.nbond = 0;
	for(i = 0; i < top.bondCount; i++){
		if ((top.bonds[i].func == 40) || (top.bonds[i].c0 == 1 && top.bonds[i].func == 10)){
			psf.nbond ++;
		}
	}
	psf.bonds = (PSFBond*)calloc(psf.nbond, sizeof(PSFBond));
	int currentBond = 0;
	for(i = 0; i < top.bondCount; i++){
		if ((top.bonds[i].func == 40) || (top.bonds[i].c0 == 1 && top.bonds[i].func == 10)){
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
		mdd.h_coord[i].x = xyz.atoms[i].x/10.0;		// [angstr] -> [nm]
		mdd.h_coord[i].y = xyz.atoms[i].y/10.0;		// [angstr] -> [nm]
		mdd.h_coord[i].z = xyz.atoms[i].z/10.0;		// [angstr] -> [nm]		
	}
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

void MDGPU::init()
{
	initTimer();
	
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
		//TODO
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

	// TODO Add reading fixatoms from PDB


	int* fixedAtomsMask;
	fixedAtomsMask = (int*)calloc(mdd.N, sizeof(int));
	if (getYesNoParameter(PARAMETER_FIX, DEFAULT_FIX)){

		PDB fixedAtomsPDB;

		getMaskedParameter(filename, PARAMETER_STRUCTURE_FILENAME);
		readPDB(filename, &fixedAtomsPDB);

		//int fix_atomType = getIntegerParameter(PARAMETER_FIX_ATOMTYPE) - 1;
		for (int i = 0; i < mdd.N; i++){
		//	if (mdd.h_atomTypes[i] == fix_atomType){
				fixedAtomsMask[i] = int(fixedAtomsPDB.atoms[i].beta);
		//	} else {
		//		fixedAtomsMask[i] = 0;
		//	}
		}
	}

	char integ_str[PARAMETER_MAX_LENGTH];
	getMaskedParameter(integ_str, PARAMETER_INTEGRATOR);

	if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG) == 0) {
		integrator = new LeapFrog(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NEW) == 0) { //TODO CHANGE NEW, add to parameters.h
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		integrator = new LeapFrog_new(&mdd, temperature, seed);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_OVERDUMPED) == 0) { //add to parameters.h
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		integrator = new LeapFrog_overdumped(&mdd, temperature, seed, fixedAtomsMask); // TODO add random force
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_VELOCITY_VERLET) == 0) {
		integrator = new VelocityVerlet(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER) == 0) {
		float tau = getFloatParameter(PARAMETER_NOSE_HOOVER_TAU);
		float T0 = getFloatParameter(PARAMETER_NOSE_HOOVER_T0);
		integrator = new LeapFrogNoseHoover(&mdd, tau, T0, fixedAtomsMask);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_STEEPEST_DESCENT) == 0) { //add to parameters.h
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		float maxForce = getFloatParameter(PARAMETER_STEEPEST_DESCENT_MAXFORCE);
		integrator = new SteepestDescent(&mdd, temperature, seed, fixedAtomsMask, maxForce);
	} else {
		DIE("Integrator was set incorrectly!");
	}

	if(getYesNoParameter(PARAMETER_LANGEVIN, DEFAULT_LANGEVIN)){
		float damping = getFloatParameter(PARAMETER_DAMPING);
		int seed = getIntegerParameter(PARAMETER_LANGEVIN_SEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		potentials.push_back(new Langevin(&mdd, damping, seed, temperature));
	}

//DNA	
	//BondsClass2Atom

	int bondCountsPar = par.bondCount;
	int bondCountsTop = 0;

	for(i = 0; i < top.bondCount; i++){
		if(top.bonds[i].func == 10){
			bondCountsTop++;
		}
	}

	int4* pair;
	pair = (int4*)calloc(bondCountsTop, sizeof(int4));

	bondCountsTop = 0;
	for(i = 0; i < top.bondCount; i++){
		if(top.bonds[i].func == 10){
			pair[bondCountsTop].x = top.bonds[i].i - 1;
			pair[bondCountsTop].y = top.bonds[i].j - 1;
			pair[bondCountsTop].z = (int)top.bonds[i].c0 - 1;
			bondCountsTop++;
		}
	}

	float4* bondCoeffs;
	bondCoeffs = (float4*)calloc(bondCountsPar, sizeof(float4));

	for(int i = 0; i < bondCountsPar; i++){
		bondCoeffs[i].x = par.bondCoeff[i].l0/10.0;		// [angstr] -> [nm]
		bondCoeffs[i].y = par.bondCoeff[i].k2*4.184*100.0;	// [kcal/(mol*angstr^2)] -> [kJ/(mol*nm^2)]
		bondCoeffs[i].z = par.bondCoeff[i].k3*4.184*1000.0;	// [kcal/(mol*angstr^3)] -> [kJ/(mol*nm^3)]
		bondCoeffs[i].w = par.bondCoeff[i].k4*4.184*10000.0;	// [kcal/(mol*angstr^4)] -> [kJ/(mol*nm^4)]
	}

	potentials.push_back(new BondsClass2Atom(&mdd, bondCountsPar, bondCountsTop, pair, bondCoeffs));

	//AngleClass2

	int angleCountsPar = par.angleCount;
	int angleCountsTop = 0;

	for(i = 0; i < top.angleCount; i++){
		if(top.angles[i].func == 10){
			angleCountsTop++;
		}
	}

	int4* angle;
	angle = (int4*)calloc(angleCountsTop, sizeof(int4));

	angleCountsTop = 0;
	for(i = 0; i < top.angleCount; i++){
		if(top.angles[i].func == 10){
			angle[i].x = top.angles[i].i - 1;
			angle[i].y = top.angles[i].j - 1;
			angle[i].z = top.angles[i].k - 1;
			angle[i].w = (int)top.angles[i].c0 - 1;
			angleCountsTop++;
		}
	}

	float4* angleCoeffs;
	angleCoeffs = (float4*)calloc(angleCountsPar, sizeof(float4));

	for(int i = 0; i < angleCountsPar; i++){
		angleCoeffs[i].x = par.angleCoeff[i].theta0;	// [degree]
		angleCoeffs[i].y = par.angleCoeff[i].k2*4.184;	// [kcal/(mol*rad^2)] -> [kJ/(mol*rad^2)]
		angleCoeffs[i].z = par.angleCoeff[i].k3*4.184;	// [kcal/(mol*rad^3)] -> [kJ/(mol*rad^3)]
		angleCoeffs[i].w = par.angleCoeff[i].k4*4.184;	// [kcal/(mol*rad^4)] -> [kJ/(mol*rad^4)]
	}

	checkCUDAError("before Angle");
	potentials.push_back(new AngleClass2(&mdd, angleCountsPar, angleCountsTop, angle, angleCoeffs));
	checkCUDAError("after Angle");


//PROTEIN

	if(getYesNoParameter(PARAMETER_PROTEIN, DEFAULT_PROTEIN)){

		//FENE
		int bondCount = 0;
		for (i = 0; i < top.bondCount; i++){
			if (top.bonds[i].func == 40){						//TODO
				bondCount++;
			}
		}

		int2* bondsFENE;
		bondsFENE = (int2*)calloc(bondCount, sizeof(int2));
		
		float* bondsFENE_C0;
		bondsFENE_C0 = (float*)calloc(bondCount, sizeof(float));

		bondCount = 0;
		for (int b = 0; b < top.bondCount; b++){
			if(top.bonds[b].func == 40){						//TODO
				bondsFENE[bondCount].x = top.bonds[b].i - 1;
				bondsFENE[bondCount].y = top.bonds[b].j - 1;
				bondsFENE_C0[bondCount] = top.bonds[b].c0/10.0;			// [angstr] -> [nm]
				bondCount++;
			}
		}

		checkCUDAError("before FENE");
		potentials.push_back(new FENE(&mdd, bondCount, bondsFENE, bondsFENE_C0));
		checkCUDAError("after FENE");

//=================================================================================================================
		//LJP
		int pairCount = 0;
		for (i = 0; i < top.pairsCount; i++){
			if(top.pairs[i].func == 40){						//TODO
				pairCount++;
			}
		}

		int2* pairsLJP;
		pairsLJP = (int2*)calloc(pairCount, sizeof(int2));

		float* pairsLJP_C0;
		pairsLJP_C0 = (float*)calloc(pairCount, sizeof(float));			// state of equilibrium

		float* pairsLJP_C1;
		pairsLJP_C1 = (float*)calloc(pairCount, sizeof(float));			// epsilon

		pairCount = 0;
		for (int b = 0; b < top.pairsCount; b++){
			if (top.pairs[b].func == 40){						//TODO
				pairsLJP[pairCount].x = top.pairs[b].i - 1;
				pairsLJP[pairCount].y = top.pairs[b].j - 1;
				pairsLJP_C0[pairCount] = top.pairs[b].c0/10.0;		// r0 [angstr] -> [nm]
				pairsLJP_C1[pairCount] = top.pairs[b].c1*4.184;		// eps [KCal/mol] -> [KJ/mol]
				pairCount++;
			}
		}

		checkCUDAError("before LJP");
		potentials.push_back(new LJP(&mdd, pairCount, pairsLJP, pairsLJP_C0, pairsLJP_C1));
		checkCUDAError("after LJP");
	}

	//Init pair lists

	float gausExclCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
	float coulCutoff = getFloatParameter(PARAMETER_COULOMB_CUTOFF);
	float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);

	int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
	int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

	std::vector<int2> exclusions(top.exclusionCount);
	for (int i = 0; i < top.exclusionCount; i++){
		if (top.exclusions[i].i < top.exclusions[i].j){
			exclusions[i].x = top.exclusions[i].i - 1;
			exclusions[i].y = top.exclusions[i].j - 1;
		} else {
			exclusions[i].x = top.exclusions[i].j - 1;
			exclusions[i].y = top.exclusions[i].i - 1;
		}
	}

	if (getYesNoParameter(PARAMETER_PROTEIN, DEFAULT_PROTEIN)){
		for (int b = 0; b < top.bondCount; b++){
			if (top.bonds[b].func == 40){
				int2 excl;
				excl.x = top.bonds[b].i - 1;
				excl.y = top.bonds[b].j - 1;
				if (excl.x > excl.y){
					int temp = excl.x;
					excl.x = excl.y;
					excl.y = temp;
				}
				exclusions.push_back(excl);
			}
		}
		for (int b = 0; b < top.pairsCount; b++){
			if (top.pairs[b].func == 40){
				int2 excl;
				excl.x = top.pairs[b].i - 1;
				excl.y = top.pairs[b].j - 1;
				if (excl.x > excl.y){
					int temp = excl.x;
					excl.x = excl.y;
					excl.y = temp;
				}
				exclusions.push_back(excl);
			}
		}
	}

	std::sort(exclusions.begin(), exclusions.end(), &int2_comparatorEx);

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

	for (int i = 1; i < top.atomCount; i++){
		if(typeCount < atoi(top.atoms[i].type)){
			typeCount = atoi(top.atoms[i].type);
		}
	}
	printf("typeCount = %d\n", typeCount);
	
	GaussExCoeff* gaussExCoeff;
	gaussExCoeff = (GaussExCoeff*)calloc(typeCount*typeCount, sizeof(GaussExCoeff));

	for(int i = 0; i < typeCount; i++){
		for(int j = 0; j < typeCount; j++){
			for(int k = 0; k < par.ljCount; k++){
				if((i == par.lj_RepulsiveCoeff[k].i - 1 && j == par.lj_RepulsiveCoeff[k].j - 1) || (j == par.lj_RepulsiveCoeff[k].i - 1 && i == par.lj_RepulsiveCoeff[k].j - 1)){
					gaussExCoeff[i+j*typeCount].l = par.lj_RepulsiveCoeff[k].l;
					gaussExCoeff[i+j*typeCount].A = par.lj_RepulsiveCoeff[k].A*4.184/pow(10.0, gaussExCoeff[i+j*typeCount].l);	// [kcal/mol*angstr^l] -> [kJ/mol*nm^l];
				}
			}
			for(int k = 0; k < par.gaussCount; k++){
				if((i == par.gaussCoeff[k].i - 1 && j == par.gaussCoeff[k].j - 1) || (j == par.gaussCoeff[k].i - 1 && i == par.gaussCoeff[k].j - 1)){
					gaussExCoeff[i+j*typeCount].numberGaussians = par.gaussCoeff[k].numberGaussians;
					gaussExCoeff[i+j*typeCount].B = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					gaussExCoeff[i+j*typeCount].C = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					gaussExCoeff[i+j*typeCount].R = (float*)calloc(par.gaussCoeff[k].numberGaussians, sizeof(float));
					for(int l = 0; l < par.gaussCoeff[k].numberGaussians; l++){
						gaussExCoeff[i+j*typeCount].B[l] = par.gaussCoeff[k].B[l]*4.184;		// [kcal/mol] -> [kJ/mol]
						gaussExCoeff[i+j*typeCount].C[l] = par.gaussCoeff[k].C[l]*100.0;		// [1/angstr^2] -> [1/nm^2]
						gaussExCoeff[i+j*typeCount].R[l] = par.gaussCoeff[k].R[l]/10.0;			// [angstr] -> [nm]
					}
				}		
			}
		}
	}
	
	float cutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);

	checkCUDAError("before Gauss");
	potentials.push_back(new GaussExcluded(&mdd, cutoff, typeCount, gaussExCoeff, plistL2));
	checkCUDAError("after Gauss");

	float dielectric = getFloatParameter(PARAMETER_DIELECTRIC, DEFAULT_DIELECTRIC);
	PPPM* pppm = new PPPM(&mdd, dielectric, coulCutoff);

	checkCUDAError("before PPPM");
	potentials.push_back(pppm);
	checkCUDAError("after PPPM");

	checkCUDAError("before Coulomb");
	potentials.push_back(new Coulomb(&mdd, plistL2, pppm->get_alpha(), dielectric, coulCutoff));
	checkCUDAError("after Coulomb");

	//REPULSIVE
/*
	std::vector<int2> exclusions_prot;

	for(int b = 0; b < top.bondCount; b++){
		if(top.bonds[b].func == 40){				//TODO
			int2 excl;
			excl.x = top.bonds[b].i;
			excl.y = top.bonds[b].j;
			if (excl.x > excl.y){
				int temp = excl.x;
				excl.x = excl.y;
				excl.y = temp;
				printf("WARNING\n");
			}
			exclusions_prot.push_back(excl);
		}
	}
	for(int b = 0; b < top.pairsCount; b++){
		if(top.pairs[b].func == 40){				//TODO
			int2 excl;
			excl.x = top.pairs[b].i;
			excl.y = top.pairs[b].j;
			if (excl.x > excl.y){
				int temp = excl.x;
				excl.x = excl.y;
				excl.y = temp;
				printf("WARNING\n");
			}
			exclusions_prot.push_back(excl);
		}
	}

	std::sort(exclusions_prot.begin(), exclusions_prot.end(), &int2_comparatorEx);

	//for pairList1
	//float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);
	//int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
	//for pairList2
	//float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	//int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

	float nbCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);

	PairListL1* plistL1_prot = new PairListL1(&mdd, exclusions_prot, possiblePairsCutoff, pairsCutoff, possiblePairsFreq);
	PairListL2* plistL2_prot = new PairListL2(&mdd, plistL1_prot->d_pairs, pairsCutoff, nbCutoff, pairsFreq);
	updaters.push_back(plistL1_prot);
	updaters.push_back(plistL2_prot);

	float rep_eps = getFloatParameter(PARAMETER_REPULSIVE_EPSILON);
	float rep_sigm = getFloatParameter(PARAMETER_REPULSIVE_SIGMA);

	potentials.push_back(new Repulsive(&mdd, plistL2_prot, nbCutoff, rep_eps, rep_sigm));
*/

	//PushingSphere

	if(getYesNoParameter(PARAMETER_PUSHING_SPHERE, DEFAULT_PUSHING_SPHERE)){
		float psR0 = getFloatParameter(PARAMETER_PUSHING_SPHERE_RADIUS0);
		float psR = getFloatParameter(PARAMETER_PUSHING_SPHERE_RADIUS);
		float4 pscenterPoint;
		getVectorParameter(PARAMETER_PUSHING_SPHERE_CENTER_POINT, &pscenterPoint.x, &pscenterPoint.y, &pscenterPoint.z);
		float psUpdate = getIntegerParameter(PARAMETER_PUSHING_SPHERE_UPDATE_FREQ);
		float psSigma = getFloatParameter(PARAMETER_PUSHING_SPHERE_SIGMA);
		float psEpsilon = getFloatParameter(PARAMETER_PUSHING_SPHERE_EPSILON);
		char psfilename[1024];
		getMaskedParameter(psfilename, PARAMETER_PUSHING_SPHERE_OUTPUT_FILENAME); 
		potentials.push_back(new PushingSphere(&mdd, psR0, psR, pscenterPoint, psUpdate, psSigma, psEpsilon, psfilename));
	}

	//INDENTATION

	if(getYesNoParameter(PARAMETER_INDENTATION, DEFAULT_INDENTATION)){

		int atomCount = 0;
		for (i = 0; i < top.atomCount; i++){
			if (strcmp(top.atoms[i].type, "4") == 0){
				atomCount++;
			}
		}
		printf("Indentation atomCount = %d\n", atomCount);

		float ind_tip_radius = getFloatParameter(PARAMETER_INDENTATION_TIP_RADIUS);
		float3 ind_tip_coord;
		getVectorParameter(PARAMETER_INDENTATION_TIP_COORD, &ind_tip_coord.x, &ind_tip_coord.y, &ind_tip_coord.z);
		float3 ind_base_coord;
		getVectorParameter(PARAMETER_INDENTATION_BASE_COORD, &ind_base_coord.x, &ind_base_coord.y, &ind_base_coord.z);
		int ind_base_freq = getIntegerParameter(PARAMETER_INDENTATION_BASE_DISPLACEMENT_FREQUENCY);
		float3 ind_n;
		getVectorParameter(PARAMETER_INDENTATION_N, &ind_n.x, &ind_n.y, &ind_n.z);
		float ind_vel = getFloatParameter(PARAMETER_INDENTATION_VELOCITY);

		ind_vel = (float(ind_base_freq)/60.86)*ind_vel;
		printf("vel = %f\n", ind_vel);	

		float ind_ks = getFloatParameter(PARAMETER_INDENTATION_KSPRING);
		float ind_eps = getFloatParameter(PARAMETER_INDENTATION_EPSILON);
		float ind_sigm = getFloatParameter(PARAMETER_INDENTATION_SIGMA);

	//SURFACE
		float3 sf_coord;
		getVectorParameter(PARAMETER_SURFACE_COORD, &sf_coord.x, &sf_coord.y, &sf_coord.z);
		float3 sf_n;
		getVectorParameter(PARAMETER_SURFACE_N, &sf_n.x, &sf_n.y, &sf_n.z);
		float sf_eps = getFloatParameter(PARAMETER_SURFACE_EPSILON);
		float sf_sigm = getFloatParameter(PARAMETER_SURFACE_SIGMA);


		int dcd_freq = getIntegerParameter(PARAMETER_DCD_OUTPUT_FREQUENCY);
		char pdb_cant_filename[FILENAME_LENGTH];
		getMaskedParameter(pdb_cant_filename, PARAMETER_PDB_CANTILEVER_OUTPUT_FILENAME);
		char dcd_cant_filename[FILENAME_LENGTH];
		getMaskedParameter(dcd_cant_filename, PARAMETER_DCD_CANTILEVER_OUTPUT_FILENAME);


	//PDB_CANTILEVER
		PDB pdb_cant;
		readPDB(pdb_cant_filename, &pdb_cant);

/*
		int atomCount_cant = 2;

		pdb_cant.atomCount = atomCount_cant;
		pdb_cant.atoms = (PDBAtom*)calloc(pdb_cant.atomCount, sizeof(PDBAtom));
		//tip
		pdb_cant.atoms[0].id = 1;
		strcpy(pdb_cant.atoms[0].name, "TIP");
		pdb_cant.atoms[0].chain = 'T';
		strcpy(pdb_cant.atoms[0].resName, "tip");
		pdb_cant.atoms[0].altLoc = ' ';
		pdb_cant.atoms[0].resid = 0;
		pdb_cant.atoms[0].x = ind_tip_coord.x*10.0;	// [nm] -> [angstr]
		pdb_cant.atoms[0].y = ind_tip_coord.y*10.0;	// [nm] -> [angstr]
		pdb_cant.atoms[0].z = ind_tip_coord.z*10.0;	// [nm] -> [angstr]
		//base
		pdb_cant.atoms[1].id = 2;
		strcpy(pdb_cant.atoms[1].name, "BASE");
		pdb_cant.atoms[1].chain = 'B';
		strcpy(pdb_cant.atoms[1].resName, "bas");
		pdb_cant.atoms[1].altLoc = ' ';
		pdb_cant.atoms[1].resid = 0;
		pdb_cant.atoms[1].x = ind_base_coord.x*10.0;	// [nm] -> [angstr]
		pdb_cant.atoms[1].y = ind_base_coord.y*10.0;	// [nm] -> [angstr]
		pdb_cant.atoms[1].z = ind_base_coord.z*10.0;	// [nm] -> [angstr]

		writePDB(pdb_cant_filename, &pdb_cant);
*/

		checkCUDAError("before Indentation");
		potentials.push_back(new Indentation(&mdd, atomCount, ind_tip_radius, ind_tip_coord, ind_base_coord, ind_base_freq, ind_n, ind_vel, ind_ks, ind_eps, ind_sigm, sf_coord, sf_n, sf_eps, sf_sigm, dcd_freq, dcd_cant_filename));
		checkCUDAError("after Indentation");
	}


//UPDATERS
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

void MDGPU::compute()
{
	mdd.step = 0;
	printTime(mdd.step);
	int numsteps = mdd.numsteps;
	int nav = numsteps;
	int i;
	int u, p;
	for(u = 0; u != updaters.size(); u++){
		if(nav > updaters[u]->getFrequence()){
			nav = updaters[u]->getFrequence();
			//cudaThreadSynchronize();
		}
	}

	//cudaThreadSynchronize();
	for(p = 0; p != potentials.size(); p++){
		potentials[p]->compute();
//checkCUDAError("Here!");
//printf("bla!");
		//cudaThreadSynchronize();
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

	//for(mdd.step = 0; mdd.step <= numsteps; mdd.step += nav){
	while(mdd.step <= numsteps){
		for(u = 0; u != updaters.size(); u++){
			if(mdd.step % updaters[u]->getFrequence() == 0){
				updaters[u]->update();
				//cudaThreadSynchronize();
			}
		}
		for(i = 0; i < nav; i++){
			integrator->integrate_step_one();
			//cudaThreadSynchronize();
			for(p = 0; p != potentials.size(); p++){
				potentials[p]->compute();
				//cudaThreadSynchronize();
				checkCUDAError("CUDA ERROR");
			}

			integrator->integrate_step_two();
			mdd.step ++;
		}
	}
	
	//XYZ-File ending coord
	if(getYesNoParameter(PARAMETER_OUTPUT_XYZ, DEFAULT_OUTPUT_XYZ)){
		char filename[FILENAME_LENGTH];
		getMaskedParameter(filename, PARAMETER_OUTPUT_XYZ_FILENAME);
		FILE * file;
		file = fopen(filename, "w");
		fprintf(file, "%d\n", mdd.N);
		fprintf(file, "Created by mdd.cu\n");
		for(i = 0; i < mdd.N; i++){
			fprintf(file, "%s\t%f\t%f\t%f\t", "P", mdd.h_coord[i].x, mdd.h_coord[i].y, mdd.h_coord[i].z);
		}
		fclose(file);
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

