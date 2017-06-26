/*
 * VelocityVerletLangevin.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class VelocityVerletLangevin : public IIntegrator {
public:
	VelocityVerletLangevin(MDData *mdd);
	~VelocityVerletLangevin();
	void integrate_step_one (MDData *mdd);
	void integrate_step_two (MDData *mdd);
private:
	float gamma;
	float T;
	float var;
	float4* h_oldForces;
	float4* d_oldForces;
};
