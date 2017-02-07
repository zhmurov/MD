/*
 * LeapFrogNoseHoover.cuh
 *
 *  Created on: 19.12.2013
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrogNoseHoover : public IIntegrator {
public:
	LeapFrogNoseHoover(MDData *mdd, float tau, float T0, fix_atomType);
	~LeapFrogNoseHoover();
	void integrate_step_one ();
	void integrate_step_two ();
private:
	float* h_T;
	float* d_T;
	float gamma;
	float T0;
	float tau;
	Reduction* reduction;
};
