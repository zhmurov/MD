#pragma once

#include "../Updaters/PairlistUpdater.cuh"

class Repulsive : public IPotential {
public:
	Repulsive(MDData *mdd, PairlistUpdater *plist, float nbCutoff, float rep_eps, float rep_sigm);
	~Repulsive();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Repulsive";}
	float getEnergies(int energyId, int timestep);
private:
	PairlistUpdater* plist;
	float nbCutoff;
	float rep_eps;
	float rep_sigm;
	float* h_energy;
	float* d_energy;
};
