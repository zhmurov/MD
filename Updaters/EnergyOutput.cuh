/*
 * EnergyOutput.cuh
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

#define ENERGY_OUTPUT_WIDTH			16
#define ENERGY_OUTPUT_STEP			"Step"
#define ENERGY_OUTPUT_TEMPERATURE	"Temperature"
#define ENERGY_OUTPUT_TOTAL			"Total"

class EnergyOutput : public IUpdater {
public:
	EnergyOutput(MDData *mdd, std::vector<IPotential*>* potentials);
	~EnergyOutput();
	void update(MDData *mdd);
private:
	std::vector<IPotential*>* potentials;
	char filename[FILENAME_LENGTH];
};
