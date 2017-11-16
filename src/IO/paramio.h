/*
 * paramio.h
 *
 *  Created on: July 6, 2016
 *      Author: kir_min
 */

#pragma once

#define BOND_NAME_TYPE 16 
#define ANGLE_NAME_TYPE 16 

typedef struct {
	int id;
	char typeName[BOND_NAME_TYPE];
	float l0;
	float k2;
	float k3;
	float k4;
} BondCoeff;

typedef struct {
	int id;
	char typeName[ANGLE_NAME_TYPE];
	float theta0;
	float k2;
	float k3;
	float k4;
} AngleCoeff;

typedef struct {
	int i;
	int j;
	int numberGaussians;
	float* B;
	float* C;
	float* R;
} GaussCoeff;

typedef struct {
	int i;
	int j;
	float A;
	int l;
} LJ_RepulsiveCoeff;

typedef struct {
	int bondCount;
	int angleCount;
	int gaussCount;
	int ljCount;
	BondCoeff* bondCoeff;
	AngleCoeff* angleCoeff;
	GaussCoeff* gaussCoeff;
	LJ_RepulsiveCoeff* lj_RepulsiveCoeff;
} PARAMData;

extern int readPARAM(char* filename, PARAMData* paramData);
