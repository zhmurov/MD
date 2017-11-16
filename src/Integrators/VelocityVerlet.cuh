/*
 * VelocityVerlet.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class VelocityVerlet : public IIntegrator {
public:
	VelocityVerlet(MDData *mdd, int* h_fixAtoms);
	~VelocityVerlet();
	void integrateStepOne();
	void integrateStepTwo();
private:
	int* h_fixAtoms;
	int* d_fixAtoms;
};
