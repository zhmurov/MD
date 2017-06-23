/*
 * LeapFrog.cuh
 *
 *  Created on: 22.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrog : public IIntegrator {
public:
	LeapFrog(MDData *mdd, int* h_fixAtoms);
	~LeapFrog();
	void integrateStepOne();
	void integrateStepTwo();
private:
	int* h_fixAtoms;
	int* d_fixAtoms;
};
