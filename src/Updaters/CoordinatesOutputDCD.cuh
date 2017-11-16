/*
 * CoordinatesOutputDCD.cuh
 *
 *  Created on: 16.08.2012
 *      Author: zhmurov
 */

#pragma once

#include "IO/dcdio.h"

class CoordinatesOutputDCD : public IUpdater {
public:
	CoordinatesOutputDCD(MDData *mdd, int freq, char* filename);
	~CoordinatesOutputDCD();
	void update();
private:
	DCD dcd;
};
