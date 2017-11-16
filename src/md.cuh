/*
 * md.cuh
 *
 *  Created on: 15.08.2012
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */

#pragma once

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "parameters.h"
#include "IO/configreader.h"
#include "Common/interfaces.h"
#include "IO/topio.h"
#include "IO/paramio.h"
#include "IO/psfio.h"
#include "IO/xyzio.h"
#include "Util/wrapper.h"
#include "Util/timer.h"
#include "Util/ran2.h"
#include "cuda.h"

bool int2_comparatorEx(int2 i, int2 j){
	if(i.x < j.x){
		return true;
	} else
	if(i.x == j.x){
		return i.y < j.y;
	} else {
		return false;
	}
}

class MDGPU
{

public:
	MDGPU() {};
	~MDGPU();
	void init();
	void generateVelocities(float T, int * rseed);
	void compute();

private:
	MDData mdd;
	std::vector<IPotential*> potentials;
	std::vector<IUpdater*> updaters;
	IIntegrator* integrator;
};

__device__ __constant__ MDData c_mdd;

void initGPU();
