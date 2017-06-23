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
	LeapFrog_overdumped(MDData *mdd, float T, float seed, int* h_fixatoms);
	~LeapFrog_overdumped();
	void integrateStepOne();
	void integrateStepTwo();
private:
	int* h_fixatoms;
	int* d_fixatoms;

	float* h_gama;
	float* d_gama;

	float* h_var;
	float* d_var;
};
