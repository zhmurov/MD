#pragma once

#include "../IO/topio.h"

class Harmonic : public IPotential {
public:
	Harmonic(MDData *mdd, float ks, int count, int2* bonds, float* bondsR0);
	~Harmonic();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Harmonic";}
	float getEnergies(int energyId, int timestep);
private:
	float ks;

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
