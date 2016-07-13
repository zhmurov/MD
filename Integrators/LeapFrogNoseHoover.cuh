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
	LeapFrogNoseHoover(MDData *mdd, float tau, float T0);
	~LeapFrogNoseHoover();
	void integrate_step_one (MDData *mdd);
	void integrate_step_two (MDData *mdd);
private:
	float* h_T;
	float* d_T;
	float gamma;
	float T0;
	float tau;
	Reduction* reduction;
};
