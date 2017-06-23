#pragma once

#include "../IO/topio.h"

class FENE : public IPotential {
public:
	FENE(MDData *mdd, int bondCount, int2* bonds, float* bondsC0);
	~FENE();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "FENE";}
	float getEnergies(int energyId, int timestep);
private:
	int max_Nbonds;

	int* h_bondCount;
	int* h_bondMap_atom;
	float* h_bondMap_r0;

	int* d_bondCount;
	int* d_bondMap_atom;
	float* d_bondMap_r0;

	float* h_energy;
	float* d_energy;
};
