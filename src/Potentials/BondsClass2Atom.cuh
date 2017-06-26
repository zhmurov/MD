/*
 * BondsClass2Atom.cuh
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 *  Changes: 15.08.2016
 *	Author: kir_min
 */

#pragma once

#define BOND_CLASS2_STRING "class2"

class BondsClass2Atom : public IPotential {
public:
	BondsClass2Atom(MDData *mdd, int bondCount, int bondCountTop, int4* pair, float4* bondCoeffs);
	~BondsClass2Atom();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Bond";}
	float getEnergies(int energyId, int timestep);
private:

	int* h_bondCount;
	int2* h_bonds;
	float4* h_bondPar;
	float* h_energies;

	int* d_bondCount;
	int2* d_bonds;
	float4* d_bondPar;
	float* d_energies;

	int bondedCount;
	int widthTot;
	int maxBonds;
};
