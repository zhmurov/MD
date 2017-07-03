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
#include "Potentials/Pulling.cu"

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
#include "Integrators/LeapFrogLangevin.cu"
#include "Integrators/LeapFrog_new.cu"
#include "Integrators/LeapFrogOverdamped.cu"
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

	for(int i = 0; i < top.atomCount; i++){
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

	int func_fene, func_bc2a;

	func_fene = getIntegerParameter(PARAMETER_FUNCTIONTYPE_FENE, DEFAULT_FUNCTIONTYPE_FENE);
	func_bc2a = getIntegerParameter(PARAMETER_FUNCTIONTYPE_BONDSCLASS2ATOM, DEFAULT_FUNCTIONTYPE_BONDSCLASS2ATOM);

	for(int i = 0; i < top.bondCount; i++){
		if ((top.bonds[i].func == func_fene) || (top.bonds[i].c0 == 1 && top.bonds[i].func == func_bc2a)){
			psf.nbond ++;
		}
	}
	psf.bonds = (PSFBond*)calloc(psf.nbond, sizeof(PSFBond));
	int currentBond = 0;

	for(int i = 0; i < top.bondCount; i++){
		if ((top.bonds[i].func == func_fene) || (top.bonds[i].c0 == 1 && top.bonds[i].func == func_bc2a)){
			psf.bonds[currentBond].i = getIndexInTOP(top.bonds[i].i, &top) + 1;
			psf.bonds[currentBond].j = getIndexInTOP(top.bonds[i].j, &top) + 1;
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
	for(int i = 0; i < xyz.atomCount; i++){
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
	int i, j, b, p;

	initTimer();

	PDB pdbref;
	TOPData top;
	PARAMData par;

	char filename[FILENAME_LENGTH];
	getMaskedParameter(filename, PARAMETER_TOPOLOGY_FILENAME);
	readTOP(filename, &top);

	if(getYesNoParameter(PARAMETER_POTENTIAL_BONDSCLASS2ATOM, DEFAULT_POTENTIAL_BONDSCLASS2ATOM) || getYesNoParameter(PARAMETER_POTENTIAL_ANGLECLASS2, DEFAULT_POTENTIAL_ANGLECLASS2) || getYesNoParameter(PARAMETER_POTENTIAL_GAUSSEXCLUDED, DEFAULT_POTENTIAL_GAUSSEXCLUDED)){
		getMaskedParameter(filename, PARAMETER_PARAMETERS_FILENAME);
		readPARAM(filename, &par);
	}

	getMaskedParameter(filename, PARAMETER_PSF_OUTPUT_FILENAME);
	dumpPSF(filename, top);

	//TODO
	int feneFunc, ljFunc, repFunc; //protein
	int func_bc2a, func_ac2; //dna

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

	for(i = 0; i < mdd.N; i++){
		mdd.h_charge[i] = top.atoms[i].charge;
		mdd.h_atomTypes[i] = atoi(top.atoms[i].type) - 1; //TODO
	}

	for(i = 0; i < mdd.N; i++){
		mdd.h_mass[i] = top.atoms[i].mass;
	}

	float totalMass = 0.0f;
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

	int* fixedAtomsMask;
	fixedAtomsMask = (int*)calloc(mdd.N, sizeof(int));
	for(i = 0; i < mdd.N; i++){
		fixedAtomsMask[i] = 0;
	}

	if(getYesNoParameter(PARAMETER_FIX, DEFAULT_FIX) || getYesNoParameter(PARAMETER_PULLING, DEFAULT_PULLING)){

		getMaskedParameter(filename, PARAMETER_PDB_REFERENCE_FILENAME);
		readPDB(filename, &pdbref);

		if(getYesNoParameter(PARAMETER_FIX, DEFAULT_FIX)){
			for(i = 0; i < mdd.N; i++){
				fixedAtomsMask[i] = int(pdbref.atoms[i].beta);
			}
		}
	}

	char integ_str[PARAMETER_MAX_LENGTH];
	getMaskedParameter(integ_str, PARAMETER_INTEGRATOR);

	if(strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG) == 0){
		integrator = new LeapFrog(&mdd, fixedAtomsMask);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NEW) == 0){
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		integrator = new LeapFrog_new(&mdd, temperature, seed, fixedAtomsMask);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_LANGEVIN) == 0){
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		float damping = getFloatParameter(PARAMETER_LEAP_FROG_LANGEVIN_DAMPING, -1.0);
		integrator = new LeapFrogLangevin(&mdd, temperature, seed, fixedAtomsMask, damping);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_OVERDUMPED) == 0){
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		integrator = new LeapFrogOverdamped(&mdd, temperature, seed, fixedAtomsMask);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_VELOCITY_VERLET) == 0){
		integrator = new VelocityVerlet(&mdd, fixedAtomsMask);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER) == 0){
		float tau = getFloatParameter(PARAMETER_NOSE_HOOVER_TAU);
		float T0 = getFloatParameter(PARAMETER_NOSE_HOOVER_T0);
		integrator = new LeapFrogNoseHoover(&mdd, tau, T0, fixedAtomsMask);
	}else if (strcmp(integ_str, VALUE_INTEGRATOR_STEEPEST_DESCENT) == 0){
		int seed = getIntegerParameter(PARAMETER_RSEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		float maxForce = getFloatParameter(PARAMETER_STEEPEST_DESCENT_MAXFORCE);
		integrator = new SteepestDescent(&mdd, temperature, seed, maxForce, fixedAtomsMask);
	}else{
		DIE("Integrator was set incorrectly!\n");
	}

	if(getYesNoParameter(PARAMETER_LANGEVIN, DEFAULT_LANGEVIN)){
		float damping = getFloatParameter(PARAMETER_DAMPING);
		int seed = getIntegerParameter(PARAMETER_LANGEVIN_SEED);
		float temperature = getFloatParameter(PARAMETER_TEMPERATURE);
		potentials.push_back(new Langevin(&mdd, damping, seed, temperature));
	}

//=====================================================================
//DNA POTENTIALS
//=====================================================================

	//BondsClass2Atom potential
	if(getYesNoParameter(PARAMETER_POTENTIAL_BONDSCLASS2ATOM, DEFAULT_POTENTIAL_BONDSCLASS2ATOM)){

		func_bc2a = getIntegerParameter(PARAMETER_FUNCTIONTYPE_BONDSCLASS2ATOM, DEFAULT_FUNCTIONTYPE_BONDSCLASS2ATOM);

		int bondCountsPar = par.bondCount;
		int bondCountsTop = 0;

		for(b = 0; b < top.bondCount; b++){
			if(top.bonds[b].func == func_bc2a){
				bondCountsTop++;
			}
		}

		int4* pair;
		pair = (int4*)calloc(bondCountsTop, sizeof(int4));

		bondCountsTop = 0;
		for(i = 0; i < top.bondCount; i++){
			if(top.bonds[i].func == func_bc2a){
				pair[bondCountsTop].x = getIndexInTOP(top.bonds[i].i, &top);
				pair[bondCountsTop].y = getIndexInTOP(top.bonds[i].j, &top);
				pair[bondCountsTop].z = (int)top.bonds[i].c0 - 1;
				bondCountsTop++;
			}
		}

		float4* bondCoeffs;
		bondCoeffs = (float4*)calloc(bondCountsPar, sizeof(float4));

		for(i = 0; i < bondCountsPar; i++){
			bondCoeffs[i].x = par.bondCoeff[i].l0/10.0;		// [angstr] -> [nm]
			bondCoeffs[i].y = par.bondCoeff[i].k2*4.184*100.0;	// [kcal/(mol*angstr^2)] -> [kJ/(mol*nm^2)]
			bondCoeffs[i].z = par.bondCoeff[i].k3*4.184*1000.0;	// [kcal/(mol*angstr^3)] -> [kJ/(mol*nm^3)]
			bondCoeffs[i].w = par.bondCoeff[i].k4*4.184*10000.0;	// [kcal/(mol*angstr^4)] -> [kJ/(mol*nm^4)]
		}
		checkCUDAError("CUDA ERROR: before BondClass2Atom potential\n");
		potentials.push_back(new BondsClass2Atom(&mdd, bondCountsPar, bondCountsTop, pair, bondCoeffs));
		checkCUDAError("CUDA ERROR: after BondClass2Atom potential\n");
	}

	//AngleClass2 potential
	if(getYesNoParameter(PARAMETER_POTENTIAL_ANGLECLASS2, DEFAULT_POTENTIAL_ANGLECLASS2)){

		func_ac2 = getIntegerParameter(PARAMETER_FUNCTIONTYPE_ANGLECLASS2, DEFAULT_FUNCTIONTYPE_ANGLECLASS2);

		int angleCountsPar = par.angleCount;
		int angleCountsTop = 0;

		for(i = 0; i < top.angleCount; i++){
			if(top.angles[i].func == func_ac2){
				angleCountsTop++;
			}
		}

		int4* angle;
		angle = (int4*)calloc(angleCountsTop, sizeof(int4));

		angleCountsTop = 0;
		for(i = 0; i < top.angleCount; i++){
			if(top.angles[i].func == func_ac2){
				angle[i].x = getIndexInTOP(top.angles[i].i, &top);
				angle[i].y = getIndexInTOP(top.angles[i].j, &top);
				angle[i].z = getIndexInTOP(top.angles[i].k, &top);
				angle[i].w = (int)top.angles[i].c0 - 1;
				angleCountsTop++;
			}
		}

		float4* angleCoeffs;
		angleCoeffs = (float4*)calloc(angleCountsPar, sizeof(float4));

		for(i = 0; i < angleCountsPar; i++){
			angleCoeffs[i].x = par.angleCoeff[i].theta0;	// [degree]
			angleCoeffs[i].y = par.angleCoeff[i].k2*4.184;	// [kcal/(mol*rad^2)] -> [kJ/(mol*rad^2)]
			angleCoeffs[i].z = par.angleCoeff[i].k3*4.184;	// [kcal/(mol*rad^3)] -> [kJ/(mol*rad^3)]
			angleCoeffs[i].w = par.angleCoeff[i].k4*4.184;	// [kcal/(mol*rad^4)] -> [kJ/(mol*rad^4)]
		}

		checkCUDAError("CUDA ERROR: before AngleClass2 potential\n");
		potentials.push_back(new AngleClass2(&mdd, angleCountsPar, angleCountsTop, angle, angleCoeffs));
		checkCUDAError("CUDA ERROR: after AngleClass2 potential\n");
	}

	float dielectric = getFloatParameter(PARAMETER_DIELECTRIC, DEFAULT_DIELECTRIC);
	float coulCutoff = getFloatParameter(PARAMETER_COULOMB_CUTOFF);

	//Initialization of pairLists
	if(getYesNoParameter(PARAMETER_POTENTIAL_GAUSSEXCLUDED, DEFAULT_POTENTIAL_GAUSSEXCLUDED) || getYesNoParameter(PARAMETER_POTENTIAL_COULOMB, DEFAULT_POTENTIAL_COULOMB)){

		float gausExclCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
		float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
		float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);
		int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
		int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

		std::vector<int2> exclusions(top.exclusionCount);
		for (i = 0; i < top.exclusionCount; i++){
			if(getIndexInTOP(top.exclusions[i].i, &top) < getIndexInTOP(top.exclusions[i].j, &top)){
				exclusions[i].x = getIndexInTOP(top.exclusions[i].i, &top);
				exclusions[i].y = getIndexInTOP(top.exclusions[i].j, &top);
			} else {
				exclusions[i].x = getIndexInTOP(top.exclusions[i].j, &top);
				exclusions[i].y = getIndexInTOP(top.exclusions[i].i, &top);
			}
		}
		std::sort(exclusions.begin(), exclusions.end(), &int2_comparatorEx);

		PairListL1* plistL1 = new PairListL1(&mdd, exclusions, possiblePairsCutoff, pairsCutoff, possiblePairsFreq);
		PairListL2* plistL2 = new PairListL2(&mdd, plistL1->d_pairs, pairsCutoff, coulCutoff, pairsFreq);
		updaters.push_back(plistL1);
		updaters.push_back(plistL2);

		//GaussExcluded potential
		if(getYesNoParameter(PARAMETER_POTENTIAL_GAUSSEXCLUDED, DEFAULT_POTENTIAL_GAUSSEXCLUDED)){

			int typeCount = 1;
			bool boo;
			for (i = 1; i < top.atomCount; i++){
				for(j = 0; j < i; j++){
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

			for (i = 1; i < top.atomCount; i++){
				if(typeCount < atoi(top.atoms[i].type)){
					typeCount = atoi(top.atoms[i].type);
				}
			}
			printf("typeCount = %d\n", typeCount);
	
			GaussExCoeff* gaussExCoeff;
			gaussExCoeff = (GaussExCoeff*)calloc(typeCount*typeCount, sizeof(GaussExCoeff));

			for(i = 0; i < typeCount; i++){
				for(j = 0; j < typeCount; j++){
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

			checkCUDAError("CUDA ERROR: before GaussExcluded potential\n");
			potentials.push_back(new GaussExcluded(&mdd, cutoff, typeCount, gaussExCoeff, plistL2));
			checkCUDAError("CUDA ERROR: after GaussExcluded potential\n");
		}

		//Coulomb potential
		//PPPM potential
		if(getYesNoParameter(PARAMETER_POTENTIAL_COULOMB, DEFAULT_POTENTIAL_COULOMB) && getYesNoParameter(PARAMETER_POTENTIAL_PPPM, DEFAULT_POTENTIAL_PPPM)){

			PPPM* pppm = new PPPM(&mdd, dielectric, coulCutoff);

			checkCUDAError("CUDA ERROR: before PPPM potential\n");
			potentials.push_back(pppm);
			checkCUDAError("CUDA ERROR: after PPPM potential\n");

			checkCUDAError("CUDA ERROR: before Coulomb potential\n");
			potentials.push_back(new Coulomb(&mdd, plistL2, pppm->get_alpha(), dielectric, coulCutoff));
			checkCUDAError("CUDA ERROR: after Coulomb potential\n");
		}
	}


//=====================================================================
//PROTEIN POTENTIALS
//=====================================================================

	//FENE potential
	if(getYesNoParameter(PARAMETER_POTENTIAL_FENE, DEFAULT_POTENTIAL_FENE)){

		feneFunc = getIntegerParameter(PARAMETER_FUNCTIONTYPE_FENE, DEFAULT_FUNCTIONTYPE_FENE);
		float feneKs = getFloatParameter(PARAMETER_KS_FENE);	//spring constant					[kJ/(mol*nm^2)]
		float feneR = getFloatParameter(PARAMETER_R_FENE);	//tolerance to the change of the covalent bond length	[nm]

		int feneCount = 0;
		for(b = 0; b < top.bondCount; b++){
			if(top.bonds[b].func == feneFunc){
				feneCount++;
			}
		}

		int2* feneBonds;
		feneBonds = (int2*)calloc(feneCount, sizeof(int2));

		float* feneBondsR0;
		feneBondsR0 = (float*)calloc(feneCount, sizeof(float));

		feneCount = 0;
		for(b = 0; b < top.bondCount; b++){
			if(top.bonds[b].func == feneFunc){
				feneBonds[feneCount].x = getIndexInTOP(top.bonds[b].i, &top);
				feneBonds[feneCount].y = getIndexInTOP(top.bonds[b].j, &top);
				feneBondsR0[feneCount] = top.bonds[b].c0/10.0f; 			// [angstr]->[nm]
				feneCount++;
			}
		}

		checkCUDAError("CUDA ERROR: before FENE potential\n");
		potentials.push_back(new FENE(&mdd, feneKs, feneR, feneCount, feneBonds, feneBondsR0));
		checkCUDAError("CUDA ERROR: after FENE potential\n");
	}

	//LennardJones potential
	if(getYesNoParameter(PARAMETER_POTENTIAL_LENNARDJONES, DEFAULT_POTENTIAL_LENNARDJONES)){

		ljFunc = getIntegerParameter(PARAMETER_FUNCTIONTYPE_LENNARDJONES, DEFAULT_FUNCTIONTYPE_LENNARDJONES);

		int ljCount = 0;
		for(p = 0; p < top.pairsCount; p++){
			if(top.pairs[p].func == ljFunc){
				ljCount++;
			}
		}

		int2* ljPairs;
		ljPairs = (int2*)calloc(ljCount, sizeof(int2));

		float* ljPairsR0;
		ljPairsR0 = (float*)calloc(ljCount, sizeof(float));		// equilibrium distance

		float* ljPairsEps;
		ljPairsEps = (float*)calloc(ljCount, sizeof(float));		// epsilon

		ljCount = 0;
		for(p = 0; p < top.pairsCount; p++){
			if(top.pairs[p].func == ljFunc){
				ljPairs[ljCount].x = getIndexInTOP(top.pairs[p].i, &top);
				ljPairs[ljCount].y = getIndexInTOP(top.pairs[p].j, &top);
				ljPairsR0[ljCount] = top.pairs[p].c0/10.0; 	// [angstr]->[nm]
				ljPairsEps[ljCount] = top.pairs[p].c1*4.184; 	// [kCal/mol]->[kJ/mol]
				ljCount++;
			}
		}
		checkCUDAError("CUDA ERROR: before LennardJones potential\n");
		potentials.push_back(new LJP(&mdd, ljCount, ljPairs, ljPairsR0, ljPairsEps));
		checkCUDAError("CUDA ERROR: after LennardJones potential\n");
	}

	//Repulsive potential (PPPM and Coulumb have to be off)
	if(getYesNoParameter(PARAMETER_POTENTIAL_REPULSIVE, DEFAULT_POTENTIAL_REPULSIVE) && !getYesNoParameter(PARAMETER_POTENTIAL_GAUSSEXCLUDED, DEFAULT_POTENTIAL_GAUSSEXCLUDED) && !getYesNoParameter(PARAMETER_POTENTIAL_COULOMB, DEFAULT_POTENTIAL_COULOMB)){

		repFunc = getIntegerParameter(PARAMETER_FUNCTIONTYPE_REPULSIVE, DEFAULT_FUNCTIONTYPE_REPULSIVE);

		std::vector<int2> exclusions(top.exclusionCount);

		for(i = 0; i < top.exclusionCount; i++){
			if(top.exclusions[i].func == repFunc){
				if(getIndexInTOP(top.exclusions[i].i, &top) < getIndexInTOP(top.exclusions[i].j, &top)){
					exclusions[i].x = getIndexInTOP(top.exclusions[i].i, &top);
					exclusions[i].y = getIndexInTOP(top.exclusions[i].j, &top);
				}else{
					exclusions[i].x = getIndexInTOP(top.exclusions[i].j, &top);
					exclusions[i].y = getIndexInTOP(top.exclusions[i].i, &top);
				}
			}
		}

		std::sort(exclusions.begin(), exclusions.end(), &int2_comparatorEx);

		//pairList1
		float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);
		int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
		//pairList2
		float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
		int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

		float nbCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);

		PairListL1* plistL1 = new PairListL1(&mdd, exclusions, possiblePairsCutoff, pairsCutoff, possiblePairsFreq);
		PairListL2* plistL2 = new PairListL2(&mdd, plistL1->d_pairs, pairsCutoff, nbCutoff, pairsFreq);
		updaters.push_back(plistL1);
		updaters.push_back(plistL2);

		float repEps = getFloatParameter(PARAMETER_REPULSIVE_EPSILON);
		float repSigm = getFloatParameter(PARAMETER_REPULSIVE_SIGMA);

		checkCUDAError("CUDA ERROR: before Repulsive potential\n");
		potentials.push_back(new Repulsive(&mdd, plistL2, nbCutoff, repEps, repSigm));
		checkCUDAError("CUDA ERROR: after Repulsive potential\n");
	}

//=====================================================================
//OTHER POTENTIALS (NO DNA, NO PROTEIN)
//=====================================================================

	//PushingSphere potential
	if(getYesNoParameter(PARAMETER_PUSHING_SPHERE, DEFAULT_PUSHING_SPHERE)){

		float psR0 = getFloatParameter(PARAMETER_PUSHING_SPHERE_RADIUS0);

		float psR = getFloatParameter(PARAMETER_PUSHING_SPHERE_RADIUS, -1.0);
		float psV = getFloatParameter(PARAMETER_PUSHING_SPHERE_SPEED, 0.0);

		if((psR > 0) && (psV == 0)){
			psV = (psR0 - psR)/mdd.numsteps;
		}

		float4 pscenterPoint;
		getVectorParameter(PARAMETER_PUSHING_SPHERE_CENTER_POINT, &pscenterPoint.x, &pscenterPoint.y, &pscenterPoint.z);
		float psUpdate = getIntegerParameter(PARAMETER_PUSHING_SPHERE_UPDATE_FREQ);
		float psSigma = getFloatParameter(PARAMETER_PUSHING_SPHERE_SIGMA);
		float psEpsilon = getFloatParameter(PARAMETER_PUSHING_SPHERE_EPSILON);
		int lj_or_harmonic = 0;
		if(getYesNoParameter(PARAMETER_PUSHING_SPHERE_HARMONIC, DEFAULT_PUSHING_SPHERE_HARMONIC)){
			lj_or_harmonic = 1; 		
		}
		char psFilename[1024];
		getMaskedParameter(psFilename, PARAMETER_PUSHING_SPHERE_OUTPUT_FILENAME);
		
		int* push_mask;
		push_mask = (int*)calloc(top.atomCount, sizeof(int));

		if(getYesNoParameter(PARAMETER_PUSHING_SPHERE_MASK, DEFAULT_PUSHING_SPHERE_MASK)){
			char psPDBFilename[1024];
			getMaskedParameter(psPDBFilename, PARAMETER_PUSHING_SPHERE_MASK_PDB_FILENAME);
			PDB push_maskPDB;
			readPDB(psPDBFilename, &push_maskPDB);
			for(i = 0; i < push_maskPDB.atomCount; i++){
				if((int)push_maskPDB.atoms[i].occupancy == 1){
					push_mask[i] = 1;	
				}			
			}
		}else{
			for(i = 0; i < top.atomCount; i++){
				if(atoi(top.atoms[i].type) == 1){
					push_mask[i] = 1;
				}			
			}		
		}

		checkCUDAError("CUDA ERROR: before PushingSphere potential\n");
		potentials.push_back(new PushingSphere(&mdd, psR0, psV, pscenterPoint, psUpdate, psSigma, psEpsilon, psFilename, lj_or_harmonic, push_mask));
		checkCUDAError("CUDA ERROR: after PushingSphere potential\n");
	}

	//Pulling potential
	if(getYesNoParameter(PARAMETER_PULLING, DEFAULT_PULLING)){

		getMaskedParameter(filename, PARAMETER_PDB_REFERENCE_FILENAME);
		if(mdd.N != pdbref.atomCount){
			printf("Error: number of atoms in top is not equal the number of atoms in pdbref\n");
		}

		float3* pullBaseR0;
		pullBaseR0 = (float3*)calloc(pdbref.atomCount, sizeof(float3));
		int pullBaseFreq = getIntegerParameter(PARAMETER_PULLING_BASE_DISPLACEMENT_FREQUENCY);
		float3* pullN;
		pullN = (float3*)calloc(pdbref.atomCount, sizeof(float3));
		float pullVel = getFloatParameter(PARAMETER_PULLING_VELOCITY);
		float* pullKs;
		pullKs = (float*)calloc(pdbref.atomCount, sizeof(float));
		int dcdFreq = getIntegerParameter(PARAMETER_DCD_OUTPUT_FREQUENCY);

		//pdbref.atoms.occupancy - spring constant
		//pdbref.atoms.x(y,z) - force vector

		for(i = 0; i < pdbref.atomCount; i++){
			if(pdbref.atoms[i].occupancy != 0.0f){
				pullBaseR0[i].x = mdd.h_coord[i].x;
				pullBaseR0[i].y = mdd.h_coord[i].y;
				pullBaseR0[i].z = mdd.h_coord[i].z;

				pullN[i].x = pdbref.atoms[i].x;
				pullN[i].y = pdbref.atoms[i].y;
				pullN[i].z = pdbref.atoms[i].z;

				pullKs[i] = pdbref.atoms[i].occupancy;
			}
		}

		checkCUDAError("CUDA ERROR: before Pulling potential\n");
		potentials.push_back(new Pulling(&mdd, pullBaseR0, pullBaseFreq, pullVel, pullN, pullKs, dcdFreq));
		checkCUDAError("CUDA ERROR: after Pulling potential\n");
	}

	//Indentation potential
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

		//surface
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

		//TODO TODO TODO
		//cantilever
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

		checkCUDAError("CUDA ERROR: before Indentation potential\n");
		potentials.push_back(new Indentation(&mdd, atomCount, ind_tip_radius, ind_tip_coord, ind_base_coord, ind_base_freq, ind_n, ind_vel, ind_ks, ind_eps, ind_sigm, sf_coord, sf_n, sf_eps, sf_sigm, dcd_freq, dcd_cant_filename));
		checkCUDAError("CUDA ERROR: after Indentation potential\n");
	}


//UPDATERS
	updaters.push_back(new CoordinatesOutputDCD(&mdd));
	updaters.push_back(new EnergyOutput(&mdd, &potentials));
	
	if(getYesNoParameter(PARAMETER_FIX_MOMENTUM, DEFAULT_FIX_MOMENTUM)){
		updaters.push_back(new FixMomentum(&mdd, getIntegerParameter(PARAMETER_FIX_MOMENTUM_FREQUENCE)));
	}

	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	printf("\nMEMORY USED: %f%%\n", 100.0f*(1.0f - float(free_mem)/float(total_mem)));
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
	int u;		// updater
	int p;		// potential

	// nav - lowest updaters frequency
	for(u = 0; u != updaters.size(); u++){
		if(nav > updaters[u]->getFrequence()){
			nav = updaters[u]->getFrequence();
		}
	}
	// TODO
	for(p = 0; p != potentials.size(); p++){
		potentials[p]->compute();
	}
	while(mdd.step <= numsteps){
		for(u = 0; u != updaters.size(); u++){
			if(mdd.step % updaters[u]->getFrequence() == 0){
				updaters[u]->update();
			}
		}
		for(i = 0; i < nav; i++){
			integrator->integrateStepOne();
			for(p = 0; p != potentials.size(); p++){
				potentials[p]->compute();
				checkCUDAError("CUDA ERROR: after potential inside MDGPU:compute()\n");
			}
			integrator->integrateStepTwo();
			mdd.step++;
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
			fprintf(file, "%s\t%f\t%f\t%f\n", "P", mdd.h_coord[i].x*10.0, mdd.h_coord[i].y*10.0, mdd.h_coord[i].z*10.0);
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

