#pragma once

#include "../IO/topio.h"

class FENE : public IPotential {
public:
	FENE(MDData *mdd, int bondCount, int2* bonds, float* bondsC0);
	~FENE();
	void compute();
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) { return "FENE";}
	float get_energies(int energy_id, int timestep);
private:
	int max_pairCount;

	int* h_pairCount;
	int* h_pairMap_atom;
	float* h_pairMap_r0;

	int* d_pairCount;
	int* d_pairMap_atom;
	float* d_pairMap_r0;

	float* h_energy;
	float* d_energy;
};
