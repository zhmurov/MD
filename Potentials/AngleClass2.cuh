/*
 * AngleClass2.cuh
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
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
	AngleClass2(MDData *mdd, ReadTopology &top, ReadParameters &par);
	~AngleClass2();
	void compute(MDData *mdd);
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "Angle";}
	float get_energies(int energy_id, int timestep);
private:
	AngleData h_ad;
	AngleData d_ad;
	int angleCount;
	int widthTot;
	int lastAngled;
	int blockSizeSum;
	int blockCountSum;
};

texture<float4, 1, cudaReadModeElementType> t_anglePar;
