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
	SteepestDescent(MDData *mdd, float T, float seed, int* h_fixatoms, float maxForce);
	~SteepestDescent();
	void integrateStepOne();
	void integrateStepTwo();
private:

	float maxForce;
	
	int* h_fixatoms;
	int* d_fixatoms;

	float* h_gama;
	float* d_gama;

	float* h_var;
	float* d_var;
};
