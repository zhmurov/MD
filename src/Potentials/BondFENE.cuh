#pragma once

#include "../IO/topio.h"

class BondFENE : public IPotential {
public:
	BondFENE(MDData *mdd, float* ks, float R, int count, int2* bonds, float* bondsR0);
	~BondFENE();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "BondFENE";}
	float getEnergies(int energyId, int timestep);
private:
	float* d_ks;
	float R;

	int maxBonds;

	int* h_bondCount;
	int* d_bondCount;

	int* h_bondMap;
	int* d_bondMap;

	float* h_bondMapR0;
	float* d_bondMapR0;

	float* h_energy;
	float* d_energy;
};
