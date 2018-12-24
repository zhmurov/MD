#pragma once

#include "../IO/topio.h"

class FENE : public IPotential {
public:
	FENE(MDData *mdd, float ks, float R, int count, int2* bonds, float* bondsR0);
	~FENE();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "FENE";}
	float getEnergies(int energyId, int timestep);
private:
	float ks;
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
