#pragma once

#include "../Updaters/PairlistUpdater.cuh"

class Repulsive : public IPotential {
public:
	Repulsive(MDData *mdd, PairlistUpdater *plist, float nbCutoff, float rep_eps, float rep_sigm);
	~Repulsive();
	void compute();
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "Repulsive";}
	float get_energies(int energy_id, int timestep);
private:
	PairlistUpdater* plist;
	float nbCutoff;
	float rep_eps;
	float rep_sigm;
	float* h_energy;
	float* d_energy;
};
