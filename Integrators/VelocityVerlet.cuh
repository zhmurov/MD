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
	VelocityVerlet(MDData *mdd);
	~VelocityVerlet();
	void integrateStepOne();
	void integrateStepTwo();
private:
};
