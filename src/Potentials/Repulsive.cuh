#pragma once

#include "../Updaters/PairlistUpdater.cuh"

class Repulsive : public IPotential {
public:
	Repulsive(MDData *mdd, PairlistUpdater *plist, float nbCutoff, float eps, float sigm);
	~Repulsive();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Repulsive";}
	float getEnergies(int energyId, int timestep);
private:
	PairlistUpdater* plist;
	float nbCutoff;
	float eps;
	float sigm;

	float* h_energy;
	float* d_energy;
};
