/*
 * Langevin.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 *  Changes: 12.07.2016
 *	Author: kir_min
 */

#pragma once

class Langevin : public IPotential {
public:
	Langevin(MDData *mdd, float damping, int seed, float temperature);
	~Langevin();
	void compute();
	int getEnergyCount(){return 0;}
	std::string getEnergyName(int energyId){return "Langevin";}
	float getEnergies(int energyId, int timestep){return 0.0f;}
private:
	float var;
	float damping;
	float temperature;
};
