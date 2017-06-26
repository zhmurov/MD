/*
 * AngleClass2.cuh
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */

#pragma once

#include "math.h"

#define ANGLE_CLASS2_STRING "class2"

#define SMALL 0.0001f

class AngleClass2 : public IPotential {
public:
	AngleClass2(MDData *mdd, int angleCountPar, int angleCountTop, int4* angle, float4* angleCoeffs);
	~AngleClass2();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Angle";}
	float getEnergies(int energyId, int timestep);
private:

	int4* h_angles;
	int4* h_refs;
	float4* h_pars;
	int* h_count;
	float4* h_forces;
	float* h_energies;

	int4* d_angles;
	int4* d_refs;
	float4* d_pars;
	int* d_count;
	float4* d_forces;
	float* d_energies;

	int angleCount;
	int widthTot;
	int lastAngled;
	int blockSizeSum;
	int blockCountSum;
};

