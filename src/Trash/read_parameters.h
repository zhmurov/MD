/*
 * read_parameters.h
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <vector>
#include <string>

#include "read_topology.h"

using std::vector;
using std::string;

#define MAX_POTENTIAL_NAME_SIZE 64

struct Coeffs
{
	Coeffs() {}
	~Coeffs() {}

public:
	int id;
	string name;
	vector<string> coeffs;
};

struct Excl_Gauss_Coeffs {
  int ng;
  int l; // V_ex = A/r^l;
  double A, *B, *C, *R;
  bool ex_flag;

  Excl_Gauss_Coeffs(): ng(0), ex_flag(false) { B=C=R=NULL; }
/*  Excl_Gauss_Coeffs(int nng) {
        ng = nng;
        ex_flag = false;
        B = new double[ng];
        C = new double[ng];
        R = new double[ng];
  }*/
  ~Excl_Gauss_Coeffs() {
        if (B) delete [] B;
        if (C) delete [] C;
        if (R) delete [] R;
  }
};

class ReadParameters
{
public:
	ReadParameters(char *filename, ReadTopology *top);
	~ReadParameters();
	void read_parameters(char *filename);
	int read_coeffs(FILE *file, Coeffs *coeffs, int ncoeffs, const char *str);
	void read_gauss_coeffs(FILE *file, Excl_Gauss_Coeffs **coeffs, int natom_types);
	void read_lj_excl_coeffs(FILE *file, Excl_Gauss_Coeffs **coeffs, int natom_types);
	void allocate();
	void print_error(const char *format,...);
	char *trim(char *str);

public:
	ReadTopology *top;
	Coeffs *bond_coeffs;
	Coeffs *angle_coeffs;
	Coeffs *dihedral_coeffs;
	Coeffs *improper_coeffs;
	Excl_Gauss_Coeffs **ex_gauss_coeffs;

	int nbond_types, nangle_types, ndihedral_types, nimproper_types;

	int bond_coeffs_flag, angle_coeffs_flag, dihedral_coeffs_flag, improper_coeffs_flag;
	int gauss_coeffs_flag, lj_excl_coeffs;

	bool allocated;
};
