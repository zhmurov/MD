#pragma once

#include "../IO/topio.h"

class FENE : public IPotential {
public:
	FENE(MDData *mdd, float Ks, float R, int Count, int2* Bonds, float* BondsR0);
	~FENE();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "FENE";}
	float getEnergies(int energyId, int timestep);
private:
	float Ks;
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
