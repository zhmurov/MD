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
	void integrate_step_one (MDData *mdd);
	void integrate_step_two (MDData *mdd);
};
