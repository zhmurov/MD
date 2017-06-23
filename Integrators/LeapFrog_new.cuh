/*
 * LeapFrog.cuh
 *
 *  Created on: 22.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrog_new : public IIntegrator {
public:
	LeapFrog_new(MDData *mdd, float T, float seed, int* h_fixAtoms);
	~LeapFrog_new();
	void integrateStepOne();
	void integrateStepTwo();
private:
	int* h_fixAtoms;
	int* d_fixAtoms;

	float* h_gamma;
	float* d_gamma;

	float* h_var;
	float* d_var;
};
