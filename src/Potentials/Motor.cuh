/*
 * Motor.cu
 *
 *  Created on: 24.07.2017
 *      Author: kir_min
 */

#pragma once

#include "math.h"

class Motor : public IPotential {
public:
	Motor(MDData *mdd, float R0, float4 centerPoint, float motorForce, float radiusHole, float h, int updatefreq, float sigma, float epsilon, const char* outdatfilename, int* pushMask);
	~Motor();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "PushingSphere";}
	float getEnergies(int energyId, int timestep);
private:
	float R0;
	float motorForce;
	float radiusHole;
	float h;
	float4 centerPoint;
	int updatefreq;
	float sigma;
	float epsilon;
	int* h_mask;
	int* d_mask;
	float* h_pressureOnSphere;
	float* d_pressureOnSphere;
	char filename[1024];
	float* h_energy;
	float* d_energy;
};
