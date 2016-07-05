/*
 * GaussExcluded.cuh
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 */

#pragma once
#include "../Updaters/PairlistUpdater.cuh"

typedef struct {
	float2* exclPar;
	float3* gaussPar;
	int* gaussCount;
	float2* energies;
} GEData;

class GaussExcluded : public IPotential {
public:
	GaussExcluded(MDData *mdd, ReadTopology &top, ReadParameters &par, PairlistUpdater *pl);
	~GaussExcluded();
	void compute(MDData *mdd);
	int get_energy_count() {return 2;}
	std::string get_energy_name(int energy_id) {
		if(energy_id == 0)
			return "Gaussian";
		else
			return "Excluded";
	}
	float get_energies(int energy_id, int timestep);
private:
	PairlistUpdater *plist;
	GEData h_ged;
	GEData d_ged;
	float energyValues[2];
	int maxGaussCount;
	int atomTypesCount;
	int pdisp;
	float cutoff;
};

