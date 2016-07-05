/*
 * md.cu
 *
 *  Created on: 15.08.2012
 *      Author: zhmurov
 */

#include "md.cuh"

// Util
#include "Util/ReductionAlgorithms.cu"
#include "Util/ReductionAlgorithmsFloat4.cu"

// Potentials
#include "Potentials/BondsClass2Atom.cu"
#include "Potentials/BondsClass2Pair.cu"
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




void MDGPU::init(ReadTopology &top, ReadParameters &par)
{
	cudaSetDevice(getIntegerParameter(PARAMETER_GPU_DEVICE));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	mdd.N = top.natoms;
	mdd.widthTot = ((mdd.N-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	mdd.dt = getFloatParameter(PARAMETER_TIMESTEP);
	mdd.numsteps = getIntegerParameter(PARAMETER_NUMSTEPS);

	mdd.ftm2v = FTM2V;

	mdd.bc.rlo.x = top.box.xlo;
	mdd.bc.rlo.y = top.box.ylo;
	mdd.bc.rlo.z = top.box.zlo;

	mdd.bc.rhi.x = top.box.xhi;
	mdd.bc.rhi.y = top.box.yhi;
	mdd.bc.rhi.z = top.box.zhi;

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

	cudaMalloc((void**)&mdd.d_coord, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_vel, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_force, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_mass, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_charge, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_atomTypes, mdd.N*sizeof(int));
	cudaMalloc((void**)&mdd.d_boxids, mdd.N*sizeof(int4));


	int i, j;
	for(i = 0; i < mdd.N; i++){
		mdd.h_coord[i].x = top.atoms[i].x;
		mdd.h_coord[i].y = top.atoms[i].y;
		mdd.h_coord[i].z = top.atoms[i].z;
		mdd.h_charge[i] = top.atoms[i].charge;
		mdd.h_atomTypes[i] = top.atoms[i].type - 1;
	}

	for(i = 0; i < mdd.N; i++){
		for(j = 0; j < top.natom_types; j++){
			if(top.atoms[i].type == top.masses[j].id){
				mdd.h_mass[i] = top.masses[j].mass;
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

	// Computational arrays
	ComputationalArrays ca(&top, &par);

	std::vector<int3> bonds;
	std::vector<Coeffs> bond_coeffs;
	std::vector<int2> exclusions;

	// Add potentials, updaters and integrators

	char integ_str[PARAMETER_MAX_LENGTH];
	getMaskedParameter(integ_str, PARAMETER_INTEGRATOR);
	if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG)==0) {
		integrator = new LeapFrog(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_VELOCITY_VERLET)==0) {
		integrator = new VelocityVerlet(&mdd);
	} else if (strcmp(integ_str, VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER)==0) {
		integrator = new LeapFrogNoseHoover(&mdd);
	} else {
		DIE("Integrator was set incorrectly!");
	}

	ca.GetBondList(string(BOND_CLASS2_STRING), &bonds, &bond_coeffs);
//	potentials.push_back(new BondsClass2Atom(&mdd, bonds, bond_coeffs));
	potentials.push_back(new BondsClass2Pair(&mdd, bonds, bond_coeffs));

	potentials.push_back(new AngleClass2(&mdd, top, par));
	if(getYesNoParameter(PARAMETER_LANGEVIN, DEFAULT_LANGEVIN)){
		potentials.push_back(new Langevin(&mdd, top, par));
	}

	// Init pair lists

	float gausExclCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
	float coulCutoff = getFloatParameter(PARAMETER_COULOMB_CUTOFF);
	float pairsCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	float possiblePairsCutoff = getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);

	int possiblePairsFreq = getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);
	int pairsFreq = getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

	std::vector<int> exclTypes;
	if(hasParameter(PARAMETER_EXCLUDE_BOND_TYPES)) {
		exclTypes = getIntegerArrayParameter(PARAMETER_EXCLUDE_BOND_TYPES);
	}
	ca.GetExclusionList(&exclusions, &exclTypes);

	PairListL1* plistL1 = new PairListL1(&mdd, exclusions, possiblePairsCutoff, pairsCutoff, possiblePairsFreq);
	PairListL2* plistL2 = new PairListL2(&mdd, plistL1->d_pairs, pairsCutoff, coulCutoff, pairsFreq);
	updaters.push_back(plistL1);
	updaters.push_back(plistL2);
	if(coulCutoff - gausExclCutoff > 10.0f){
		PairListL2* plistGausExcl = new PairListL2(&mdd, plistL2->d_pairs, coulCutoff, gausExclCutoff, pairsFreq);
		potentials.push_back(new GaussExcluded(&mdd, top, par, plistGausExcl));
		updaters.push_back(plistGausExcl);
	} else {
		potentials.push_back(new GaussExcluded(&mdd, top, par, plistL2));
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

void MDGPU::compute(ReadTopology &top, ReadParameters &par)
{
	mdd.step = 0;
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
		potentials[p]->compute(&mdd);
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

	for(mdd.step = 0; mdd.step <= numsteps; mdd.step += nav){
		for(u = 0; u != updaters.size(); u++){
			if(mdd.step % updaters[u]->getFrequence() == 0){
				updaters[u]->update(&mdd);
				//cudaThreadSynchronize();
			}
		}
		for(i = 0; i < nav; i++){
			integrator->integrate_step_one(&mdd);
			//cudaThreadSynchronize();
			for(p = 0; p != potentials.size(); p++){
				potentials[p]->compute(&mdd);
				//cudaThreadSynchronize();
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

void compute(ReadTopology &top, ReadParameters &par){

	MDGPU mdgpu;
	mdgpu.init(top, par);
	mdgpu.compute(top, par);
	cudaDeviceReset();
}

