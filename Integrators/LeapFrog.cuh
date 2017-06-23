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
	LeapFrog(MDData *mdd);
	~LeapFrog();
	void integrateStepOne();
	void integrateStepTwo();
};
