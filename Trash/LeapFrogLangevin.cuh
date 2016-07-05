/*
 * LeapFrogLangevin.cuh
 *
 *  Created on: 15.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class LeapFrogLangevin : public IIntegrator {
public:
	LeapFrogLangevin(MDData *mdd);
	~LeapFrogLangevin();
	void integrate_step_one (MDData *mdd);
	void integrate_step_two (MDData *mdd);
private:
	float var;
};
