#pragma once

#include "../IO/topio.h"

class LJP : public IPotential {
public:
	LJP(MDData *mdd, int count, int2* pairs, float* pairsR0, float* pairsEps);
	~LJP();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energy_id){return "LJP";}
	float getEnergies(int energyId, int timestep);
private:
	int maxPairs;

	int* h_pairCount;
	int* d_pairCount;

	int* h_pairMap;
	int* d_pairMap;

	float* h_pairMapR0;
	float* d_pairMapR0;

	float* h_pairMapEps;
	float* d_pairMapEps;

	float* h_energy;
	float* d_energy;
};
