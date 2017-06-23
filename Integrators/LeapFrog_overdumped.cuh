/*
 * LeapFrog_overdumped.cuh
 *
 *  Created on: 11.07.2016
 *
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrog_overdumped : public IIntegrator {
public:
	LeapFrog_overdumped(MDData *mdd, float T, float seed, int* h_fixAtoms);
	~LeapFrog_overdumped();
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
