/*
 * Coulomb.cuh
 *
 *  Created on: 05.09.2012
 *      Author: zhmurov
 */

#pragma once

#include "../Updaters/PairlistUpdater.cuh"

typedef struct {
} CoulombData;

class Coulomb : public IPotential {
public:
	Coulomb(MDData *mdd, PairlistUpdater *pl, float alpha, float dielectric, float cutoff);
	~Coulomb();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Coulomb";}
	float getEnergies(int energyId, int timestep);
private:
	MDData* mdd;
	PairlistUpdater *plist;

	float* h_energy;
	float* d_energy;

	float alpha;
	float dielectric;
	float cutoff;
	float cutoffSq;
	float kc;
	float energyValue;
};

