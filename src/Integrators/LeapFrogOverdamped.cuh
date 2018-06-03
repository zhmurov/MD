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
	LeapFrogOverdamped(MDData *mdd, float temperature, float gamma, float seed, int* h_fixAtoms);
	~LeapFrogOverdamped();
	void integrateStepOne();
	void integrateStepTwo();
private:
	float gamma;
	float var;

	int* d_fixAtoms;
};
