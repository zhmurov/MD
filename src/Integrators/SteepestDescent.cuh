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
	SteepestDescent(MDData *mdd, float T, float seed, float maxForce, int* h_fixAtoms);
	~SteepestDescent();
	void integrateStepOne();
	void integrateStepTwo();
private:
	float maxForce;
	
	int* h_fixAtoms;
	int* d_fixAtoms;

	float* h_gamma;
	float* d_gamma;

	float* h_var;
	float* d_var;
};
