/*
 * GaussExcluded.cuh
 *
 *  Created on: 24.08.2012
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */

#pragma once
#include "../Updaters/PairlistUpdater.cuh"

typedef struct {
	float2* exclPar;
	float3* gaussPar;
	int* gaussCount;
	float2* energies;
} GEData;

typedef struct {
	float A;
	int l;
	int numberGaussians;
	float* B;
	float* C;
	float* R;
} GaussExCoeff;

class GaussExcluded : public IPotential {
public:
	GaussExcluded(MDData *mdd, float cutoffCONF, int typeCount, GaussExCoeff* gauss, PairlistUpdater *pl);
	~GaussExcluded();
	void compute();
	int getEnergyCount(){return 2;}
	std::string getEnergyName(int energyId) {
		if(energyId == 0)
			return "Gaussian";
		else
			return "Excluded";
	}
	float getEnergies(int energyId, int timestep);
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

