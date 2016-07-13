/*
 * md.cuh
 *
 *  Created on: 15.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "parameters.h"
#include "IO/configreader.h"
#include "Common/interfaces.h"
#include "IO/topio.h"
#include "IO/paramio.h"
#include "IO/psfio.h"
#include "IO/xyzio.h"
#include "Util/wrapper.h"
#include "Util/ran2.h"
#include "cuda.h"

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
texture<float4, 1, cudaReadModeElementType> t_coord;
texture<int, 1, cudaReadModeElementType> t_atomTypes;
texture<float, 1, cudaReadModeElementType> t_charges;

void initGPU();
