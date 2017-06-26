/*
 * read_topology.h
 *
 *  Created on: Aug 14, 2012
 *      Author: aram
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Atom
{
	Atom() : charge(0.0), x(0.0), y(0.0), z(0.0), coord_flag(false) {}
	Atom(int id, int mol_id, int res_id, int type, double charge, double xx, double yy, double zz) :
		id(id), mol_id(mol_id), res_id(res_id), type(type), charge(charge), x(xx), y(yy), z(zz), coord_flag(true) {}
	Atom(int id, int mol_id, int res_id, int type, double charge) :
		id(id), mol_id(mol_id), res_id(res_id), type(type), charge(charge), x(0.0), y(0.0), z(0.0), coord_flag(false) {}
	~Atom() {}

public:
	int id;
	int mol_id;
	int res_id;
	int type;
	double charge;
	double x, y, z;
	bool coord_flag;
};

struct Bond
{
	Bond() {}
	Bond(int id, int ty, int atm1, int atm2) : id(id), type(ty), atom1(atm1), atom2(atm2) {}
	~Bond() {}

public:
	int id;
	int type;
	int atom1;
	int atom2;
};

struct Angle
{
	Angle() {}
	Angle(int id, int ty, int atm1, int atm2, int atm3) : id(id), type(ty), atom1(atm1), atom2(atm2), atom3(atm3) {}
	~Angle() {}

public:
	int id;
	int type;
	int atom1;
	int atom2;
	int atom3;
};

struct Dihedral
{
	Dihedral() {}
	Dihedral(int id, int ty, int atm1, int atm2, int atm3, int atm4) : id(id), type(ty), atom1(atm1), atom2(atm2), atom3(atm3), atom4(atm4) {}
	~Dihedral() {}

public:
	int id;
	int type;
	int atom1;
	int atom2;
	int atom3;
	int atom4;
};

struct Improper
{
	Improper() {}
	Improper(int id, int ty, int atm1, int atm2, int atm3, int atm4) : id(id), type(ty), atom1(atm1), atom2(atm2), atom3(atm3), atom4(atm4) {}
	~Improper() {}

public:
	int id;
	int type;
	int atom1;
	int atom2;
	int atom3;
	int atom4;
};

struct Mass
{
	Mass() {}
	Mass(int id, double m) : id(id), mass(m) {}
	~Mass() {}

	int id;
	double mass;
};

struct SBox
{
	SBox() : xlo(0.0), xhi(0.0), ylo(0.0), yhi(0.0), zlo(0.0), zhi(0.0) {}
	SBox(double xlo, double xhi, double ylo, double yhi, double zlo, double zhi) :
		xlo(xlo), xhi(xhi), ylo(ylo), yhi(yhi), zlo(zlo), zhi(zhi) {}
	~SBox() {}

	double xlo, xhi;
	double ylo, yhi;
	double zlo, zhi;
};

class ReadTopology
{
public:
	ReadTopology(char *filename);
	~ReadTopology();
	void read_topology(char *filename);
	void read_atoms(FILE *file);
	void read_bonds(FILE *file);
	void read_angles(FILE *file);
	void read_dihedrals(FILE *file);
	void read_impropers(FILE *file);
	void read_masses(FILE *file);
	void allocate();
	void print_error(const char *txt);
	char *trim(char *str);

public:
	int natoms;
	int nbonds;
	int nangles;
	int ndihedrals;
	int nimpropers;
	int natom_types;
	int nbond_types;
	int nangle_types;
	int ndihedral_types;
	int nimproper_types;

	Atom *atoms;
	Bond *bonds;
	Angle *angles;
	Dihedral *dihedrals;
	Improper *impropers;
	Mass *masses;
	SBox box;

	int atoms_flag, bonds_flag, angles_flag, dihedrals_flag, impropers_flag, masses_flag;

	bool allocated;
};

