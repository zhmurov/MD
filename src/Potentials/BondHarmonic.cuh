#pragma once

#include "../IO/topio.h"

class BondHarmonic : public IPotential {
public:
	BondHarmonic(MDData *mdd, int count, int2* bonds, float* bondsR0, float* bondsKs);
	~BondHarmonic();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "BondHarmonic";}
	float getEnergies(int energyId, int timestep);
private:
	int maxBonds;

	int* h_bondCount;
	int* d_bondCount;

	int* h_bondMap;
	int* d_bondMap;

	float* h_bondMapR0;
	float* d_bondMapR0;

	float* h_bondMapKs;
	float* d_bondMapKs;

	float* h_energy;
	float* d_energy;
};
