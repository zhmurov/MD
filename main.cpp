/*
 * main.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: Aram Davtyan, Artem Zhmurov
 */

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "IO/read_topology.h"
#include "IO/read_parameters.h"
#include "IO/configreader.h"
#include "IO/psfio.h"
#include "IO/xyzio.h"
#include "parameters.h"
//#include "Util/ReductionAlgorithms.h"

using namespace std;

extern void compute(ReadTopology &top, ReadParameters &par);

void dumpPSF(char* filename, ReadTopology &top);
void readCoordinatesFromFile(char* filename, ReadTopology &top);

int main(int argc, char *argv[])
{
	parseParametersFile(argv[1], argc, argv);

	char filename[FILENAME_LENGTH];
	getMaskedParameter(filename, PARAMETER_TOPOLOGY_FILENAME);
	ReadTopology top(filename);

	getMaskedParameter(filename, PARAMETER_PARAMETERS_FILENAME);
	ReadParameters par(filename,&top);

	getMaskedParameter(filename, PARAMETER_PSF_OUTPUT_FILENAME);
	dumpPSF(filename, top);

	getMaskedParameter(filename, PARAMETER_COORDINATES_FILENAME, "NONE");

	if(strncmp(filename, "NONE", 4) != 0){
		readCoordinatesFromFile(filename, top);
	}

	compute(top, par);

	destroyConfigReader();

	return 0;
}

void dumpPSF(char* filename, ReadTopology &top){
	PSF psf;
	psf.natom = top.natoms;
	psf.ntheta = 0;
	psf.nphi = 0;
	psf.nimphi = 0;
	psf.nnb = 0;
	psf.ncmap = 0;
	psf.atoms = (PSFAtom*)calloc(psf.natom, sizeof(PSFAtom));

	int i, j;
	for(i = 0; i < top.natoms; i++){
		psf.atoms[i].id = top.atoms[i].id;
		for(j = 0; j < top.natom_types; j++){
			if(top.atoms[i].type == top.masses[j].id){
				psf.atoms[i].m = top.masses[j].mass;
			}
		}
		sprintf(psf.atoms[i].name, "C");
		sprintf(psf.atoms[i].type, "%d", top.atoms[i].type);
		psf.atoms[i].q = top.atoms[i].charge;
		if(top.atoms[i].res_id == 0){
			sprintf(psf.atoms[i].resName, "ION");
		} else {
			sprintf(psf.atoms[i].resName, "DNA");
		}
		psf.atoms[i].resid = top.atoms[i].res_id;
		sprintf(psf.atoms[i].segment, "%d", top.atoms[i].mol_id);
	}

	psf.nbond = 0;
	for(i = 0; i < top.nbonds; i++){
		if(top.bonds[i].type == 1){
			psf.nbond ++;
		}
	}
	psf.bonds = (PSFBond*)calloc(psf.nbond, sizeof(PSFBond));
	int currentBond = 0;
	for(i = 0; i < top.nbonds; i++){
		if(top.bonds[i].type == 1){
			psf.bonds[currentBond].i = top.bonds[i].atom1;
			psf.bonds[currentBond].j = top.bonds[i].atom2;
			currentBond++;
		}
	}

	writePSF(filename, &psf);
	free(psf.atoms);
	free(psf.bonds);
}

void readCoordinatesFromFile(char* filename, ReadTopology &top){
	XYZ xyz;
	readXYZ(filename, &xyz);
	int i;
	for(i = 0; i < xyz.atomCount; i++){
		top.atoms[i].x = xyz.atoms[i].x;
		top.atoms[i].y = xyz.atoms[i].y;
		top.atoms[i].z = xyz.atoms[i].z;
	}
}
