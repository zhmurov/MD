/*
 * SteepestDescent.cuh
 *
 *  Created on: 21.03.2017
 *
 */

#pragma once

#include <Common/interfaces.h>

class SteepestDescent : public IIntegrator {
public:
	SteepestDescent(MDData *mdd, float temperature, float gamma, float seed, float maxForce, int* fixAtoms);
	~SteepestDescent();
	void integrateStepOne();
	void integrateStepTwo();
private:
	float gamma;
	float var;
	float maxForce;

	int* h_fixAtoms;
	int* d_fixAtoms;
};
