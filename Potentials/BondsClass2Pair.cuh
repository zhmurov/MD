/*
 * BondClass2Pair.cuh
 *
 *  Created on: 10.09.2012
 *      Author: zhmurov
 */

#pragma once

class BondsClass2Pair : public IPotential {
public:
	BondsClass2Pair(MDData *mdd, std::vector<int3> &bonds, std::vector<Coeffs> &parameters);
	~BondsClass2Pair();
	void compute(MDData *mdd);
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "Bond";}
	float get_energies(int energy_id, int timestep);
private:
	Coeffs* getBondCoeffs(int type, ReadParameters &par);
	bool checkBondClass2(int type, ReadParameters &par);


	int3* h_bonds;
	int3* d_bonds;
	int2* h_refs;
	int2* d_refs;
	float4* h_bondPar;
	float4* d_bondPar;

	int* h_bondCount;
	int* d_bondCount;
	float4* h_forces;
	float4* d_forces;
	float* h_energies;
	float* d_energies;

	int bondCount;
	int widthTot;
	int maxBonds;
	int lastBonded;
	int blockSizeSum;
	int blockCountSum;
};

texture<float4, 1, cudaReadModeElementType> t_bondParPair;
