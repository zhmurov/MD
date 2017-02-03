#pragma once

#include "../IO/topio.h"

class LJP : public IPotential {
public:
	LJP(MDData *mdd, int pairCount, int2* pairs, float* pairs_C0, float* pairs_C1);
	~LJP();
	void compute();
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "LJP";}
	float get_energies(int energy_id, int timestep);
private:
	int max_Npairs;

	int* h_pairCount;
	int* h_pairMap_atom;
	float* h_pairMap_r0;
	float* h_pairMap_eps;

	int* d_pairCount;
	int* d_pairMap_atom;
	float* d_pairMap_r0;
	float* d_pairMap_eps;

	float* h_energy;
	float* d_energy;
};
