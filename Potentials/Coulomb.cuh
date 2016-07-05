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
	void compute(MDData *mdd);
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "Coulomb";}
	float get_energies(int energy_id, int timestep);
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

