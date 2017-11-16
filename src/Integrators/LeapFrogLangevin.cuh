/*
 * LeapFrogLangevin.cuh
 *
 *  Created on: 27.06.2017
 *      Author: kir_min
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrogLangevin : public IIntegrator {
public:
	LeapFrogLangevin(MDData *mdd, float T, float seed, int* h_fixAtoms, float damping);
	~LeapFrogLangevin();
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
