#pragma once

#include "../IO/topio.h"

class BondFENE : public IPotential {
public:
	BondFENE(MDData *mdd, float R, int count, int2* bonds, float* bondsR0, float* bondsKs);
	~BondFENE();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "BondFENE";}
	float getEnergies(int energyId, int timestep);
private:
	float R;

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
