#pragma once
#include "IO/topio.h"

class FENE : public IPotential {
public:
	FENE(MDData *mdd, TOPData* topdata);
	~FENE();
	void compute(MDData *mdd);
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) { return "FENE";}
	float get_energies(int energy_id, int timestep);
	//TODO
private:
	//float k_const = 14.0;		//kspring constant {N/m}
	//float R_const = 2.0;		//tolerance to the change of the covalent bond length.

	int bondCount;

	int* h_pairCount;
	int* h_pairMap_atom;
	float* h_pairMap_dist;
	int* d_pairCount;
	int* d_pairMap_atom;
	float* d_pairMap_dist;

	float* h_energy;
	float* d_energy;
	//TODO
};
