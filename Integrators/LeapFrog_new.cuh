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
	LeapFrog_new(MDData *mdd, float T, float seed);
	~LeapFrog_new();
	void integrate_step_one ();
	void integrate_step_two ();
private:
	float* h_gama;
	float* d_gama;

	float* h_var;
	float* d_var;
};
