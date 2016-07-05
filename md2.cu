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
#include "Potentials/Langevin.cu"
#include "Potentials/FENE.cu"

// Updaters
#include "Updaters/CoordinatesOutputDCD.cu"
#include "Updaters/EnergyOutput.cu"
#include "Updaters/FixMomentum.cu"

// Integrators
#include "Integrators/LeapFrog.cu"
#include "Integrators/VelocityVerlet.cu"
#include "Integrators/LeapFrogNoseHoover.cu"

// IO
#include "IO/configreader.h"
#include "IO/topio.h"
#include "IO/pdbio.h"

PDB pdbdata;
TOPData topdata;

void MDGPU::init(ReadTopology &top, ReadParameters &par)
{
	cudaSetDevice(getIntegerParameter(PARAMETER_GPU_DEVICE));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	char filename[FILENAME_LENGTH];

	getMaskedParameter(filename, PARAMETER_TOPOLOGY_FILENAME);
	readTOP(filename, &topdata);

	getMaskedParameter(filename, PARAMETER_COORDINATES_FILENAME);
	readPDB(filename, &pdbdata);

	// Read top and pdb files created by sop-top here
	// Get the filenames from config file

	mdd.N = pdbdata.atomCount; 	// get from top/pdb files
	mdd.widthTot = ((mdd.N-1)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;
	mdd.dt = getFloatParameter(PARAMETER_TIMESTEP);
	mdd.numsteps = getIntegerParameter(PARAMETER_NUMSTEPS);

	mdd.ftm2v = FTM2V;

	// Get xlo/hi--zlo/hi using 'getFloatParameter'
	//mdd.bc.rlo.x = top.box.xlo;
	//mdd.bc.rlo.y = top.box.ylo;
	//mdd.bc.rlo.z = top.box.zlo;

	//mdd.bc.rhi.x = top.box.xhi;
	//mdd.bc.rhi.y = top.box.yhi;
	//mdd.bc.rhi.z = top.box.zhi;

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

	cudaMalloc((void**)&mdd.d_coord, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_vel, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_force, mdd.N*sizeof(float4));
	cudaMalloc((void**)&mdd.d_mass, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_charge, mdd.N*sizeof(float));
	cudaMalloc((void**)&mdd.d_atomTypes, mdd.N*sizeof(int));
	cudaMalloc((void**)&mdd.d_boxids, mdd.N*sizeof(int4));

	int i;
	for (i = 0; i < mdd.N; i++){
		mdd.h_coord[i].x = pdbdata.atoms[i].x;
		mdd.h_coord[i].y = pdbdata.atoms[i].y;
		mdd.h_coord[i].z = pdbdata.atoms[i].z;
		mdd.h_charge[i] = topdata.atoms[i].charge;
		mdd.h_atomTypes[i] = 0;
		//TODO
		//mdd.h_atomTypes[i] = top.atoms[i].type - 1;
	}

	double totalMass = 0.0;
	for (i = 0; i < mdd.N; i++){
		mdd.h_mass[i] = topdata.atoms[i].mass;
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

	if(getYesNoParameter(PARAMETER_LANGEVIN, DEFAULT_LANGEVIN)){
		potentials.push_back(new Langevin(&mdd));
	}

	// Get the bond list from top file
	// Create new FENE potential (has to be implemented)
	
	potentials.push_back(new FENE(&mdd, &topdata));

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

