/*
 * LeapFrogOverdumped.cuh
 *
 *  Created on: 11.07.2016
 *
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrogOverdamped : public IIntegrator {
public:
	LeapFrogOverdamped(MDData *mdd, float T, float seed, int* h_fixAtoms);
	~LeapFrogOverdamped();
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
