/*
 * PushingSphere.cu
 *
 *  Created on: 17.10.2016
 *      Author: kir_min
 */

#pragma once

#include "math.h"

#define PUSHING_SPHERE_LJ		0
#define PUSHING_SPHERE_HARMONIC	1

class PushingSphere : public IPotential {
public:
	PushingSphere(MDData *mdd, float R0, float vSphere, float4 centerPoint, int updatefreq, float sigma, float epsilon, const char* outdatfilename, int ljOrHarmonic, int* pushMask);
	~PushingSphere();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "PushingSphere";}
	float getEnergies(int energyId, int timestep);
private:
	float R0;
	float vSphere;
	float radius;
	float4 centerPoint;
	int updatefreq;
	float sigma;
	float epsilon;
	int ljOrHarmonic;
	int* h_mask;
	int* d_mask;
	float* h_pressureOnSphere;
	float* d_pressureOnSphere;
	char filename[1024];
	float* h_energy;
	float* d_energy;
};
