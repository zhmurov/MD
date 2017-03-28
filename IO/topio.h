/*
 * topio.h
 *
 *  Created on: May 24, 2009
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 */

#pragma once

#define TOP_ATOMTYPE_LENGTH 16
#define TOP_ATOMNAME_LENGTH 8
#define TOP_RESNAME_LENGTH 8

#define TOP_SECTION_ATOMS		0
#define TOP_SECTION_PAIRS		1
#define TOP_SECTION_ANGLES		2
#define TOP_SECTION_DIHEDRALS	3

typedef struct {
	int id;
	char type[TOP_ATOMTYPE_LENGTH];
	int resid;
	char resName[TOP_RESNAME_LENGTH];
	char name[TOP_ATOMNAME_LENGTH];
	char chain;
	float charge;
	float mass;
} TOPAtom;

typedef struct {
	int i;
	int j;
	int func;
	float c0;
	float c1;
	float c2;
	float c3;
} TOPPair;

typedef struct {
	int i;
	int j;
	int k;
	int func;
	float c0;
	float c1;
	float c2;
	float c3;
} TOPAngle;

typedef struct {
	int i;
	int j;
	int k;
	int l;
	int func;
	float c0;
	float c1;
	float c2;
	float c3;
} TOPDihedral;

typedef struct {
	int i;
	int j;
} TOPExclusion;

typedef struct {
	int atomCount;
	int bondCount;
	int pairsCount;
	int nativesCount;
	int angleCount;
	int dihedralCount;
	int exclusionCount;
	TOPAtom* atoms;
	TOPPair* bonds;
	TOPPair* natives;
	TOPPair* pairs;
	TOPAngle* angles;
	TOPDihedral* dihedrals;
	TOPExclusion* exclusions;

	int* ids;
} TOPData;

extern int readTOP(const char* filename, TOPData* topData);
extern void writeTOP(const char* filename, TOPData* topData);
//Added 28.03.17
extern int getIndexInTOP(int nr, TOPData* topData);
