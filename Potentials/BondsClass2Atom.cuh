/*
 * BondsClass2Atom.cuh
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#pragma once

#define BOND_CLASS2_STRING "class2"

typedef struct {
	int* bondCount;
	int2* bonds;
	float4* bondPar;
	float* energies;
} BondData;

class BondsClass2Atom : public IPotential {
public:
	BondsClass2Atom(MDData *mdd, std::vector<int3> &bonds, std::vector<Coeffs> &parameters);
	~BondsClass2Atom();
	void compute(MDData *mdd);
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "Bond";}
	float get_energies(int energy_id, int timestep);
private:
	Coeffs* getBondCoeffs(int type, ReadParameters &par);
	bool checkBondClass2(int type, ReadParameters &par);
	BondData h_bd;
	BondData d_bd;
	int bondedCount;
	int widthTot;
	int maxBonds;
};

texture<float4, 1, cudaReadModeElementType> t_bondPar;
