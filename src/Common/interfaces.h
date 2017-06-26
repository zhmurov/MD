/*
 * interfaces.h
 *
 *  Created on: 29.03.2012
 *      Author: zhmurov
 */
#pragma once

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "parameters.h"

#define DEFAULT_DATA_ALLIGN	32
#define DEFAULT_BLOCK_SIZE	64
#define BOLTZMANN_CONSTANT	0.008314462		// [kJ/(K*mol)]

#define FTM2V   1.0
#define QQR2E   138.59

typedef struct{
	float3 rlo;
	float3 rhi;
	float3 len;
} Boundary;

typedef struct {

	int N;
	int widthTot;

	float M;

	int step;
	int numsteps;
	float dt;

	float ftm2v;

	Boundary bc;

	float4* h_coord;
	float4* h_vel;
	float4* h_force;
	float* h_mass;
	float* h_charge;
	int* h_atomTypes;
	int4* h_boxids;

	float4* d_coord;
	float4* d_vel;
	float4* d_force;
	float* d_mass;
	float* d_charge;
	int* d_atomTypes;
	int4* d_boxids;

} MDData;

class IPotential {
public:
	virtual ~IPotential(){}
	virtual void compute() = 0;
	virtual int getEnergyCount() = 0;
	virtual std::string getEnergyName(int energyId) = 0;
	virtual float getEnergies(int energyId, int timestep) = 0;
protected:
	MDData* mdd;
	int blockSize;
	int blockCount;
	int lastStepEnergyComputed;
};


class IIntegrator {
public:
	virtual ~IIntegrator(){}
	virtual void integrateStepOne() = 0;
	virtual void integrateStepTwo() = 0;
protected:
	MDData* mdd;
	int blockSize;
	int blockCount;
	float dt;
};

class IUpdater {
public:
	virtual ~IUpdater(){}
	virtual void update() = 0;
	int getFrequence(){return frequence;}
protected:
	MDData* mdd;
	int blockSize;
	int blockCount;
	int frequence;
};
