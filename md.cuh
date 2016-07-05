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
#include "IO/read_topology.h"
#include "IO/read_parameters.h"
#include "Util/wrapper.h"
#include "Util/ran2.h"
#include "cuda.h"

class MDGPU
{

public:
	MDGPU() {};
	~MDGPU();
	void init(ReadTopology &top, ReadParameters &par);
	void generateVelocities(float T, int * rseed);
	void compute(ReadTopology &top, ReadParameters &par);

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

void initGPU(ReadTopology &top, ReadParameters &par);
