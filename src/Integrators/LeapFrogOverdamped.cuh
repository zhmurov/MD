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
	LeapFrogOverdamped(MDData *mdd, float temperature, float gamma, float seed, int* fixAtoms);
	~LeapFrogOverdamped();
	void integrateStepOne();
	void integrateStepTwo();
private:
	float gamma;
	float var;

	int* h_fixAtoms;
	int* d_fixAtoms;

	float* h_gamma;
	float* d_gamma;

	float* h_var;
	float* d_var;
};
