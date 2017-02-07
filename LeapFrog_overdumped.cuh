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
	LeapFrog_overdumped(MDData *mdd, float T, float seed);
	~LeapFrog_overdumped();
	void integrate_step_one();
	void integrate_step_two();
private:
	float* h_gama;
	float* d_gama;

	float* h_var;
	float* d_var;
};
