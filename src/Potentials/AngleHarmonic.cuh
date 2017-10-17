/*
 * AngleHarmonic.cuh
 *
 *  Created on: 15.10.2017
 *      Author: zhmurov
 */

#pragma once

#define ANGLE_HARMONIC_STRING "harmonic"

class AngleHarmonic : public IPotential {
public:
	AngleHarmonic(MDData *mdd, int angleCount, int3* angles, float2* angleParameters);
	~AngleHarmonic();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Angle";}
	float getEnergies(int energyId, int timestep);
private:

	int4* h_angles;
	float2* h_pars;
	int4* h_refs;
	int* h_count;
	float4* h_forces;
	float* h_energies;

	int4* d_angles;
	float2* d_pars;
	int4* d_refs;
	int* d_count;
	float4* d_forces;
	float* d_energies;

	int angleCount;
	int widthTot;
	int lastAngled;
	int blockSizeSum;
	int blockCountSum;
};

