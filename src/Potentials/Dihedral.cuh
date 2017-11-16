/*
 * Dihedral.cuh
 *
 *  Created on: 17.10.2017
 *      Author: zhmurov
 */

#pragma once

#define DIHEDRAL_PROPER_STRING "dihedral"

class Dihedral : public IPotential {
public:
	Dihedral(MDData *mdd, int dihedralCount, int4* dihedrals, float3* dihedralParameters);
	~Dihedral();
	void compute();
	int getEnergyCount(){return 2;}
	std::string getEnergyName(int energyId){if(energyId == 0) {return "Dihedral";} else {return "Improper";}}
	float getEnergies(int energyId, int timestep);
private:

	int4* h_dihedrals;
	float3* h_pars;
	int4* h_refs;
	int* h_count;
	float4* h_forces;
	float* h_energies;

	int4* d_dihedrals;
	float3* d_pars;
	int4* d_refs;
	int* d_count;
	float4* d_forces;
	float* d_energies;

	int dihedralCount;
	int widthTot;
	int lastDihedraled;
	int blockSizeSum;
	int blockCountSum;

	int lastStepEnergyComputed;
	float energyProper;
	float energyImproper;

};

