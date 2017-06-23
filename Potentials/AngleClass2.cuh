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

typedef struct {
	int4* angles;
	int4* refs;
	float4* pars;
	int* count;
	float4* forces;
	float* energies;
} AngleData;

class AngleClass2 : public IPotential {
public:
	AngleClass2(MDData *mdd, int angleCountPar, int angleCountTop, int4* angle, float4* angleCoeffs);
	~AngleClass2();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Angle";}
	float getEnergies(int energyId, int timestep);
private:
	AngleData h_ad;
	AngleData d_ad;
	int angleCount;
	int widthTot;
	int lastAngled;
	int blockSizeSum;
	int blockCountSum;
};

