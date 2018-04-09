//gcc pairs_to_bonds.cpp ../../../IO/topio.cpp ../../../Util/wrapper.cpp

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../../IO/topio.h"

#define BUF_SIZE 				256
#define INPUT_TOP				"3j6f_CG.top"
#define OUTPUT_TOP				"3j6f_CG1.top"

TOPData top;
TOPData newtop;

int main(){

	readTOP(INPUT_TOP, &top);

	// newTOP
	newtop.atoms = (TOPAtom*)calloc(top.atomCount, sizeof(TOPAtom));
	newtop.atomCount = top.atomCount;

	newtop.bonds = (TOPPair*)calloc((top.bondCount + top.pairsCount), sizeof(TOPPair));
	newtop.bondCount = top.bondCount + top.pairsCount;
	newtop.pairs = (TOPPair*)calloc(0, sizeof(TOPPair));
	newtop.pairsCount = 0;

	newtop.exclusions = (TOPExclusion*)calloc(top.exclusionCount, sizeof(TOPExclusion));
	newtop.exclusionCount = top.exclusionCount;

	for(int i = 0; i < newtop.atomCount; i++){
		newtop.atoms[i].id = top.atoms[i].id;
		strcpy(newtop.atoms[i].type, top.atoms[i].type);
		newtop.atoms[i].resid = top.atoms[i].resid;
		strcpy(newtop.atoms[i].resName, top.atoms[i].resName);
		strcpy(newtop.atoms[i].name, top.atoms[i].name);
		newtop.atoms[i].chain = top.atoms[i].chain;
		newtop.atoms[i].charge = top.atoms[i].charge;
		newtop.atoms[i].mass = top.atoms[i].mass;
	}

	for(int b = 0; b < newtop.bondCount; b++){
		if(b < top.bondCount){
			newtop.bonds[b].i = top.bonds[b].i;
			newtop.bonds[b].j = top.bonds[b].j;
			newtop.bonds[b].func = top.bonds[b].func;
			newtop.bonds[b].c0 = top.bonds[b].c0;						// [angstr]
			newtop.bonds[b].c1 = top.bonds[b].c1;						// [kJ/(mol*nm^2)]
		}else{
			newtop.bonds[b].i = top.pairs[b - top.bondCount].i;
			newtop.bonds[b].j = top.pairs[b - top.bondCount].j;
			newtop.bonds[b].func = top.pairs[b - top.bondCount].func;
			newtop.bonds[b].c0 = top.pairs[b - top.bondCount].c0;		// [angstr]
			newtop.bonds[b].c1 = top.pairs[b - top.bondCount].c1;		// [kJ/(mol*nm^2)]
		}
	}

	for(int e = 0; e < newtop.exclusionCount; e++){
		newtop.exclusions[e].i = top.exclusions[e].i;
		newtop.exclusions[e].j = top.exclusions[e].j;
		newtop.exclusions[e].func = top.exclusions[e].func;
	}

	writeTOP(OUTPUT_TOP, &newtop);

	return 0;
}
