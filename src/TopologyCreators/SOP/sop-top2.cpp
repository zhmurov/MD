/*
 * main.cpp
 *
 *  Created on: Nov 28, 2012
 *      Author: kias
 */

#include "Util/aatocg.h"
#include "../../IO/pdbio.h"
#include "../../IO/xyzio.h"
#include "../../IO/topio.h"
#include "../../IO/psfio.h"
#include "../../IO/configreader.h"

#define BUF_SIZE 1024

#define PARAMETER_CGCONFIG			"cgconfig"
#define PARAMETER_STRUCTURE			"structure"
#define PARAMETER_COORDINATES		"coordinates"
#define PARAMETER_ADDITIONAL_BONDS	"additional_bonds"
#define PARAMETER_COORDINATES_PDB	"coordinates_pdb"
#define PARAMETER_COORDINATES_XYZ	"coordinates_xyz"
#define PARAMETER_TOPOLOGY			"topology"
#define PARAMETER_TOPOLOGY_PSF		"topology_psf"
#define PARAMETER_TOPOLOGY_NATPSF	"topology_natpsf"
#define PARAMETER_R_LIMIT_BOND		"R_limit_bond"
#define PARAMETER_SC_LIMIT_BOND		"SC_limit_bond"
#define PARAMETER_EH				"eh"
#define PARAMETER_KS				"Ks"
#define DEFAULT_KS					8370
#define PARAMETER_USE_CHAINS		"use_chains"
#define DEFAULT_USE_CHAINS			0
#define PARAMETER_XRAY_PDB			"xraypdb"
#define PARAMETER_ELASTIC_NETWORK	"elastic_network"
#define DEFAULT_ELASTIC_NETWORK		0

CGConfig conf;
PDB pdb;

PDB xraypdb;
bool elastic_network;

struct SOPBead {
	char name[5], type[5], resname[5], segment[5];
	int resid;
	double x, y, z;
	double m, q;
	double occupancy, beta;
	std::vector<Atom> connectedTo;
	std::vector<PDBAtom> represents;
};

typedef struct {
	int i;
	int j;
	float Kb;
	float b0;
} SOPBond;

typedef struct {
	int i;
	int j;
	float eh;
	float r0;
} SOPNative;

typedef struct {
	int i;
	int j;
} SOPPair;

typedef struct {
	char res_name[5];
	int resid;
	std::vector<PDBAtom*> atoms;
}  PDBTreeResidue;

typedef struct {
	char segment_name[5];
	std::vector<PDBTreeResidue> residues;
} PDBTreeSegment;

std::vector<PDBTreeSegment> pdb_tree;
std::vector<SOPBead> beads;


bool bond_comparator(SOPBond b1, SOPBond b2){
	if(b1.i < b2.i){
		return true;
	} else
	if(b1.i == b2.i){
		return b1.j < b2.j;
	} else {
		return false;
	}
}

bool native_comparator(SOPNative n1, SOPNative n2){
	if(n1.i < n2.i){
		return true;
	} else
	if(n1.i == n2.i){
		return n1.j < n2.j;
	} else {
		return false;
	}
}

bool pairs_comparator(SOPPair p1, SOPPair p2){
	if(p1.i < p2.i){
		return true;
	} else
	if(p1.i == p2.i){
		return p1.j < p2.j;
	} else {
		return false;
	}
}

void trimString(char* string){
	//printf("'%s' -> ", string);
	int i = 0;
	int j = 0;
	while(string[j] == ' '){
		j++;
	}
	for(i = 0; i < strlen(string)-j; i++){
		string[i] = string[i+j];
	}
	for(i = strlen(string)-j; i < strlen(string); i++){
		string[i] = ' ';
	}
	i = 0;
	while(string[i] != ' ' && i < strlen(string)){
		i++;
	}
	string[i] = '\0';
	//printf("'%s'\n", string);
}

std::vector<SOPBond> bonds;
std::vector<SOPNative> natives;

std::vector<SOPPair> bonds13;
std::vector<SOPPair> exclusions;

int** bondsList;
int* bondCount;

void createBondsList(int N){
	bondsList = (int**)calloc(N, sizeof(int*));
	bondCount = (int*)calloc(N, sizeof(int));
	int i;
	for(i = 0; i < N; i++){
		bondsList[i] = (int*)calloc(10, sizeof(int));
	}
	for(i = 0; i < bonds.size(); i++){
		SOPBond bond = bonds.at(i);
		bondsList[bond.i][bondCount[bond.i]] = bond.j;
		bondsList[bond.j][bondCount[bond.j]] = bond.i;
		bondCount[bond.i] ++;
		bondCount[bond.j] ++;
	}
}

int** bonds13List;
int* bond13Count;

void createBonds13List(int N){
	bonds13List = (int**)calloc(N, sizeof(int*));
	bond13Count = (int*)calloc(N, sizeof(int));
	int i;
	for(i = 0; i < N; i++){
		bonds13List[i] = (int*)calloc(100, sizeof(int));
	}
	for(i = 0; i < bonds13.size(); i++){
		SOPPair bond = bonds13.at(i);
		bonds13List[bond.i][bond13Count[bond.i]] = bond.j;
		bonds13List[bond.j][bond13Count[bond.j]] = bond.i;
		bond13Count[bond.i] ++;
		bond13Count[bond.j] ++;
	}
}

void addConnection(char* segment1, int resid1, char* bead1,
		char* segment2, int resid2, char* bead2);
unsigned findBead(const char* segment, int resid, const char* bead_name);
void addConnection(SOPBead bead, Atom conn, unsigned int j);

void addBond(int i, int j, float Kb, float b0);
void addNative(int i, int j, float eh, float r0);
int checkBonded(int i, int j);
int check13Bonded(int i, int j);
int checkNatived(int i, int j);
bool checkIfInXRay(int beadID);
double getDistance(SOPBead bead1, SOPBead bead2);
double getDistance(PDBAtom atom1, PDBAtom atom2);


int main(int argc, char* argv[]){

	/*char string[1024];
	sprintf(string, "  A11  ");
	trimString(string);
	sprintf(string, " A11  ");
	trimString(string);
	sprintf(string, "A11  ");
	trimString(string);
	sprintf(string, "  A11 ");
	trimString(string);
	sprintf(string, "  A11");
	trimString(string);
	sprintf(string, " A11S  ");
	trimString(string);
	sprintf(string, "A11S  ");
	trimString(string);
	sprintf(string, "NZ");
	trimString(string);
	exit(0);*/


	printf("==========================\n");
	printf("SOP-GPU Topology creator 2.0\n");
	printf("==========================\n");

	int i, b;
	unsigned int j, k, l, m, n, o;

	if(argc < 1){
		printf("ERROR: Configuration file should be specified.\n");
		exit(-1);
	}

	parseParametersFile(argv[1]);

	char filename[1024];
	getMaskedParameter(filename, PARAMETER_CGCONFIG);
	readCGConfiguration_pol(filename, &conf);

	getMaskedParameter(filename, PARAMETER_STRUCTURE);
	readPDB(filename, &pdb);

	getMaskedParameter(filename, PARAMETER_COORDINATES, "NONE");
	if(strncmp(filename, "NONE", 4) != 0){
		printf("Taking coordinates from XYZ file '%s' instead of initial PDB.\n", filename);
		XYZ xyz;
		readXYZ(filename, &xyz);
		if(xyz.atomCount != pdb.atomCount){
			printf("Atom counts in provided XYZ and PDB files are not the same. Aborting.\n");
			exit(0);
		}
		for(i = 0; i < xyz.atomCount; i++){
			pdb.atoms[i].x = xyz.atoms[i].x;
			pdb.atoms[i].y = xyz.atoms[i].y;
			pdb.atoms[i].z = xyz.atoms[i].z;
		}
	}

	if(getYesNoParameter(PARAMETER_ELASTIC_NETWORK, DEFAULT_ELASTIC_NETWORK)){
		printf("Elastic network model will be created.\n");
		elastic_network = true;
	} else {
		printf("SOP model will be created.\n");
		elastic_network = false;
	}

	getMaskedParameter(filename, PARAMETER_XRAY_PDB, "NONE");
	if(strncmp(filename, "NONE", 4) != 0){
		readPDB(filename, &xraypdb);
	} else {
		xraypdb.atomCount = -1;
	}

	printf("Building PDB tree...\n");

	if(getYesNoParameter(PARAMETER_USE_CHAINS, DEFAULT_USE_CHAINS)){
		printf("Using chain entries to separate polypeptide chains.\n");
		for(i = 0; i < pdb.atomCount; i++){
			sprintf(pdb.atoms[i].segment, "%c", pdb.atoms[i].chain);
			trimString(pdb.atoms[i].segment);
		}
	} else {
		printf("Using segment entries to separate polypeptide chains.\n");
		for(i = 0; i < pdb.atomCount; i++){
			trimString(pdb.atoms[i].segment);
		}
		/*for(i = 0; i < pdb.atomCount; i++){
			printf("'%s'-", pdb.atoms[i].segment);
			sprintf(pdb.atoms[i].segment, "%s", pdb.atoms[i].segment);
			printf("'%s'\n", pdb.atoms[i].segment);
		}*/
	}
	for(i = 0; i < pdb.atomCount; i++){
		trimString(pdb.atoms[i].name);
		trimString(pdb.atoms[i].resName);
	}
	for(i = 0; i < conf.residues.size(); i++){
		trimString(conf.residues.at(i).resname);
		for(j = 0; j < conf.residues.at(i).beads.size(); j++){
			trimString(conf.residues.at(i).beads.at(j).name);
			trimString(conf.residues.at(i).beads.at(j).resname);
			trimString(conf.residues.at(i).beads.at(j).type);
		}
	}
	for(i = 0; i < pdb.atomCount; i++){
		bool found = false;
		for(j = 0; j < pdb_tree.size(); j++){
			if(strcmp(pdb_tree.at(j).segment_name, pdb.atoms[i].segment) == 0){
				found = true;
			}
		}

		if(!found){
			PDBTreeSegment segment;
			sprintf(segment.segment_name, "%s", pdb.atoms[i].segment);
			pdb_tree.push_back(segment);
		}
	}
	printf("Found %ld segments.\n", pdb_tree.size());
	for(i = 0; i < pdb.atomCount; i++){
		for(j = 0; j < pdb_tree.size(); j++){
			if(strcmp(pdb_tree.at(j).segment_name, pdb.atoms[i].segment) == 0){
				bool found = false;
				for(k = 0; k < pdb_tree.at(j).residues.size(); k++){
					if(pdb_tree.at(j).residues.at(k).resid == pdb.atoms[i].resid){
						found = true;
					}
				}
				if(!found){
					PDBTreeResidue residue;
					sprintf(residue.res_name, "%s", pdb.atoms[i].resName);
					residue.resid = pdb.atoms[i].resid;
					pdb_tree.at(j).residues.push_back(residue);
				}
			}
		}
	}
	for(i = 0; i < pdb.atomCount; i++){
		for(j = 0; j < pdb_tree.size(); j++){
			if(strcmp(pdb_tree.at(j).segment_name, pdb.atoms[i].segment) == 0){
				for(k = 0; k < pdb_tree.at(j).residues.size(); k++){
					if(pdb_tree.at(j).residues.at(k).resid == pdb.atoms[i].resid){
						pdb_tree.at(j).residues.at(k).atoms.push_back(&pdb.atoms[i]);
					}
				}
			}
		}
	}

	for(j = 0; j < pdb_tree.size(); j++){
		printf("\n\nSegment %s:\n", pdb_tree.at(j).segment_name);
		for(k = 0; k < pdb_tree.at(j).residues.size(); k++){
			printf("\nResid %s%d:\n",
					pdb_tree.at(j).residues.at(k).res_name, pdb_tree.at(j).residues.at(k).resid);
			for(l = 0; l < pdb_tree.at(j).residues.at(k).atoms.size(); l++){
				printAtom(*pdb_tree.at(j).residues.at(k).atoms.at(l));
			}
		}
	}

	printf("PDB tree completed.\n");

	printf("Creating coarse-grained beads.\n");

	for(j = 0; j < pdb_tree.size(); j++){
		for(k = 0; k < pdb_tree.at(j).residues.size(); k++){
			Residue resid;
			PDBTreeResidue pdb_tree_resid;
			for(m = 0; m < conf.residues.size(); m++){
				if(strcmp(conf.residues.at(m).resname, pdb_tree.at(j).residues.at(k).res_name) == 0){
					resid = conf.residues.at(m);
					pdb_tree_resid = pdb_tree.at(j).residues.at(k);
				}
			}
			for(n = 0; n < resid.beads.size(); n++){
				SOPBead bead;
				bead.m = resid.beads.at(n).mass;
				bead.q = resid.beads.at(n).charge;
				sprintf(bead.name, "%s", resid.beads.at(n).name);
				sprintf(bead.type, "%s", resid.beads.at(n).type);
				sprintf(bead.resname, "%s", pdb_tree_resid.res_name);
				sprintf(bead.segment, "%s", pdb_tree.at(j).segment_name);
				bead.resid = pdb_tree_resid.resid;
				bead.x = 0.0;
				bead.y = 0.0;
				bead.z = 0.0;
				bead.connectedTo = resid.beads.at(n).connectedTo;
				int count = 0;
				for(l = 0; l < pdb_tree_resid.atoms.size(); l++){
					for(o = 0; o < resid.beads.at(n).atomsCM.size(); o++){
						if(strcmp(pdb_tree_resid.atoms.at(l)->name,
								resid.beads.at(n).atomsCM.at(o).name) == 0){
							bead.x += pdb_tree_resid.atoms.at(l)->x;
							bead.y += pdb_tree_resid.atoms.at(l)->y;
							bead.z += pdb_tree_resid.atoms.at(l)->z;
							count ++;
						}
					}
				}
				if(count != 0){
					bead.x /= (double)count;
					bead.y /= (double)count;
					bead.z /= (double)count;
				    count = 0;
				    bead.beta = 0.0;
				    bead.occupancy = 0.0;
				    for(l = 0; l < pdb_tree_resid.atoms.size(); l++){
					    for(o = 0; o < resid.beads.at(n).atomsRepresents.size(); o++){
						    if(strcmp(pdb_tree_resid.atoms.at(l)->name,
								    resid.beads.at(n).atomsRepresents.at(o).name) == 0){
							    //PDBAtom atom;
							    //memcpy(&atom, &pdb_tree_resid.atoms.at(l), sizeof(PDBAtom));
							    bead.represents.push_back(*pdb_tree_resid.atoms.at(l));
							    bead.beta += pdb_tree_resid.atoms.at(l)->beta;
							    bead.occupancy += pdb_tree_resid.atoms.at(l)->occupancy;
							    count ++;
						    }
					    }
				    }
				    bead.beta /= count;
				    bead.occupancy /= count;
				    beads.push_back(bead);
                } else {
                    printf("WARNING: No coordinates for bead %s in residue %s-%s%d. Bead was not added.\n", resid.beads.at(n).name, pdb_tree.at(j).segment_name, pdb_tree_resid.res_name, pdb_tree_resid.resid);
                }
			}
		}
	}

	printf("Coarse-grained beads are added.\n");

	if(!elastic_network){
		printf("Adding covalent bonds...\n");

		for(j = 0; j < beads.size(); j++){
			SOPBead bead = beads.at(j);
			for(k = 0; k < bead.connectedTo.size(); k++){
				Atom conn = bead.connectedTo.at(k);
				addConnection(bead, conn, j);
			}
		}

		getMaskedParameter(filename, PARAMETER_ADDITIONAL_BONDS, "NONE");
		if(strncmp(filename, "NONE", 4) != 0){
			FILE* file = fopen(filename, "r");
			char* segment1;
			int resid1;
			char* bead1;
			char* segment2;
			int resid2;
			char* bead2;
			if(file != NULL){
				char* pch;
				char buffer[BUF_SIZE];
				while(fgets(buffer, BUF_SIZE, file) != NULL){
					if(strncmp(buffer, "CONN", 4) == 0){
						pch = strtok(buffer, " \t\n\r");
						pch = strtok(NULL, " \t\n\r");
						segment1 = pch;
						pch = strtok(NULL, " \t\n\r");
						resid1 = atoi(pch);
						pch = strtok(NULL, " \t\n\r");
						bead1 = pch;
						pch = strtok(NULL, " \t\n\r");
						segment2 = pch;
						pch = strtok(NULL, " \t\n\r");
						resid2 = atoi(pch);
						pch = strtok(NULL, " \t\n\r");
						bead2 = pch;
						trimString(segment1);
						trimString(bead1);
						trimString(segment2);
						trimString(bead2);
						addConnection(segment1, resid1, bead1, segment2, resid2, bead2);
					}
				}
			}
			fclose(file);
		} else {
			printf("No additional covalent bonds (S-S, crosslinks, etc.) have been specified.\n");
		}

		std::sort(bonds.begin(), bonds.end(), bond_comparator);

		createBondsList(beads.size());

		int b1, b2;
		for(i = 0; i < beads.size(); i++){
			for(b1 = 0; b1 < bondCount[i]; b1++){
				j = bondsList[i][b1];
				for(b2 = 0; b2 < bondCount[j]; b2++){
					k = bondsList[j][b2];
					if(i < k){
						SOPPair bond13;
						bond13.i = i;
						bond13.j = k;
						bonds13.push_back(bond13);
						//printf("%d-%d-%d\n", i, j, k);
					}
				}
			}
		}

		std::sort(bonds13.begin(), bonds13.end(), pairs_comparator);

		createBonds13List(beads.size());

		printf("Covalent bonds are added.\n");

		printf("Adding native contacts...\n");

		

		float eh, cutoff, cutoffAtomistic;
		char ehstring[1024];
		getMaskedParameter(ehstring, PARAMETER_EH);
		cutoff = getFloatParameter(PARAMETER_R_LIMIT_BOND);
		cutoffAtomistic = getFloatParameter(PARAMETER_SC_LIMIT_BOND);
		for(j = 0; j < beads.size(); j++){
			if(j % 100 == 0){
				printf("Bead %d out of %ld\n", j, beads.size());
			}
			for(k = j + 1; k < beads.size(); k++){
				double r0 = getDistance(beads.at(j), beads.at(k));

				if(strcmp(ehstring, "O") == 0){
					eh = sqrt(beads.at(j).occupancy*beads.at(k).occupancy);
				} else if(strcmp(ehstring, "B") == 0){
					eh = sqrt(beads.at(j).beta*beads.at(k).beta);
				} else if(atof(ehstring) != 0){
					eh = atof(ehstring);
					//printf("%f\n", eh);
				} else {
					exit(0);
				}
				bool added = false;
				if(r0 < cutoff){
					if((!checkBonded(j, k)) && (!check13Bonded(j, k)) && checkIfInXRay(j) && checkIfInXRay(k)){
						addNative(j, k, eh, r0);
						added = true;
					}
				}
				if(!added && cutoffAtomistic > 0){
					for(l = 0; l < beads.at(j).represents.size(); l++){
						for(m = 0; m < beads.at(k).represents.size(); m++){
							double r1 = getDistance(beads.at(j).represents.at(l), beads.at(k).represents.at(m));
							if((!added) && r1 < cutoffAtomistic){
								if((!checkBonded(j, k)) && (!check13Bonded(j, k)) && checkIfInXRay(j) && checkIfInXRay(k)){
									addNative(j, k, eh, r0);
									added = true;
								}
							}
						}
					}
				}
			}
		}

		printf("Native contacts are added.\n");

		std::sort(natives.begin(), natives.end(), native_comparator);


		printf("Creating exclusions list.\n");

		for(b = 0; b < bonds.size(); b++){
			SOPPair pair;
			pair.i = bonds.at(b).i;
			pair.j = bonds.at(b).j;
			exclusions.push_back(pair);
		}

		for(n = 0; n < natives.size(); n++){
			SOPPair pair;
			pair.i = natives.at(n).i;
			pair.j = natives.at(n).j;
			exclusions.push_back(pair);
		}

		std::sort(exclusions.begin(), exclusions.end(), pairs_comparator);


		double pot = 0.0;
		for(j = 0; j < natives.size(); j++){
			SOPNative native = natives.at(j);
			pot += native.eh;
		}
		printf("Total energy: %f\n", pot);
	} else {
		printf("Creating elastic network\n");

		float cutoff = getFloatParameter(PARAMETER_R_LIMIT_BOND);
		float ks = getFloatParameter(PARAMETER_KS, DEFAULT_KS);

		for(j = 0; j < beads.size(); j++){
			if(j % 100 == 0){
				printf("Bead %d out of %ld\n", j, beads.size());
			}
			for(k = j + 1; k < beads.size(); k++){
				double r0 = getDistance(beads.at(j), beads.at(k));
				if(r0 < cutoff){
					addBond(j, k, ks, r0);
				}
			}
		}
		std::sort(bonds.begin(), bonds.end(), bond_comparator);
	}

	printf("Saving PDB, PSF and TOP files for coarse-grained system...\n");

	PDB cgpdb;
	cgpdb.atomCount = beads.size();
	cgpdb.atoms = (PDBAtom*)calloc(cgpdb.atomCount, sizeof(PDBAtom));
	for(j = 0; j < beads.size(); j++){
		cgpdb.atoms[j].altLoc = ' ';
		cgpdb.atoms[j].beta = beads.at(j).m;
		cgpdb.atoms[j].chain = beads.at(j).segment[0];
		cgpdb.atoms[j].id = j+1;
		sprintf(cgpdb.atoms[j].name, "%s", beads.at(j).name);
		cgpdb.atoms[j].occupancy = beads.at(j).q;
		sprintf(cgpdb.atoms[j].resName, "%s", beads.at(j).resname);
		sprintf(cgpdb.atoms[j].segment, "%s", beads.at(j).segment);
		cgpdb.atoms[j].resid = beads.at(j).resid;
		cgpdb.atoms[j].x = beads.at(j).x;
		cgpdb.atoms[j].y = beads.at(j).y;
		cgpdb.atoms[j].z = beads.at(j).z;
	}
	getMaskedParameter(filename, PARAMETER_COORDINATES_PDB);
	writePDB(filename, &cgpdb);

	XYZ cgxyz;
	cgxyz.atomCount = beads.size();
	cgxyz.atoms = (XYZAtom*)calloc(cgxyz.atomCount, sizeof(XYZAtom));
	for(j = 0; j < beads.size(); j++){
		cgxyz.atoms[j].name = beads.at(j).name[0];
		cgxyz.atoms[j].x = beads.at(j).x;
		cgxyz.atoms[j].y = beads.at(j).y;
		cgxyz.atoms[j].z = beads.at(j).z;
	}
	getMaskedParameter(filename, PARAMETER_COORDINATES_XYZ);
	writeXYZ(filename, &cgxyz);


	getMaskedParameter(filename, PARAMETER_TOPOLOGY_PSF, "NONE");
	if(strncmp(filename, "NONE", 4) != 0){
		PSF cgpsf;
		cgpsf.natom = cgpdb.atomCount;
		cgpsf.atoms = (PSFAtom*)calloc(cgpsf.natom, sizeof(PSFAtom));
		cgpsf.nbond = 0;
		cgpsf.nbond = bonds.size();
		cgpsf.bonds = (PSFBond*)calloc(cgpsf.nbond, sizeof(PSFBond));
		cgpsf.ncmap = 0;
		cgpsf.nimphi = 0;
		cgpsf.nnb = 0;
		cgpsf.nphi = 0;
		cgpsf.ntheta = 0;
		for(i = 0; i < cgpdb.atomCount; i++){
			cgpsf.atoms[i].id = i+1;
			sprintf(cgpsf.atoms[i].name, "%s", cgpdb.atoms[i].name);
			sprintf(cgpsf.atoms[i].type, "%s", cgpdb.atoms[i].name);
			sprintf(cgpsf.atoms[i].segment, "%s", cgpdb.atoms[i].segment);

			cgpsf.atoms[i].m = cgpdb.atoms[i].beta;
			cgpsf.atoms[i].q = cgpdb.atoms[i].occupancy;


			cgpsf.atoms[i].resid = cgpdb.atoms[i].resid;
			sprintf(cgpsf.atoms[i].resName, "%s", cgpdb.atoms[i].resName);
		}
		for(b = 0; b < (int)bonds.size(); b++){
			cgpsf.bonds[b].i = bonds.at(b).i + 1;
			cgpsf.bonds[b].j = bonds.at(b).j + 1;
		}
		writePSF(filename, &cgpsf);
	}

	getMaskedParameter(filename, PARAMETER_TOPOLOGY_NATPSF, "NONE");
	if(strncmp(filename, "NONE", 4) != 0){
		PSF cgpsfnat;
		cgpsfnat.natom = cgpdb.atomCount;
		cgpsfnat.atoms = (PSFAtom*)calloc(cgpsfnat.natom, sizeof(PSFAtom));
		cgpsfnat.nbond = 0;
		cgpsfnat.nbond = natives.size();
		cgpsfnat.bonds = (PSFBond*)calloc(cgpsfnat.nbond, sizeof(PSFBond));
		cgpsfnat.ncmap = 0;
		cgpsfnat.nimphi = 0;
		cgpsfnat.nnb = 0;
		cgpsfnat.nphi = 0;
		cgpsfnat.ntheta = 0;
		for(i = 0; i < cgpdb.atomCount; i++){
			cgpsfnat.atoms[i].id = i+1;
			sprintf(cgpsfnat.atoms[i].name, "%s", cgpdb.atoms[i].name);
			sprintf(cgpsfnat.atoms[i].type, "%s", cgpdb.atoms[i].name);
			sprintf(cgpsfnat.atoms[i].segment, "%s", cgpdb.atoms[i].segment);

			cgpsfnat.atoms[i].m = cgpdb.atoms[i].beta;
			cgpsfnat.atoms[i].q = cgpdb.atoms[i].occupancy;


			cgpsfnat.atoms[i].resid = cgpdb.atoms[i].resid;
			sprintf(cgpsfnat.atoms[i].resName, "%s", cgpdb.atoms[i].resName);
		}
		for(b = 0; b < (int)natives.size(); b++){
			cgpsfnat.bonds[b].i = natives.at(b).i + 1;
			cgpsfnat.bonds[b].j = natives.at(b).j + 1;
		}
		writePSF(filename, &cgpsfnat);
	}

	TOPData cgtop;
	cgtop.atomCount = cgpdb.atomCount;
	cgtop.bondCount = bonds.size();
	cgtop.pairsCount = natives.size();
	cgtop.exclusionCount = exclusions.size();
	cgtop.angleCount = 0;
	cgtop.dihedralCount = 0;
	cgtop.atoms = (TOPAtom*)calloc(cgtop.atomCount, sizeof(TOPAtom));
	cgtop.bonds = (TOPPair*)calloc(cgtop.bondCount, sizeof(TOPPair));
	cgtop.pairs = (TOPPair*)calloc(cgtop.pairsCount, sizeof(TOPPair));
	cgtop.exclusions = (TOPExclusion*)calloc(cgtop.exclusionCount, sizeof(TOPExclusion));
	for(i = 0; i < cgpdb.atomCount; i++){
		cgtop.atoms[i].id = i;
		sprintf(cgtop.atoms[i].name, "%s", cgpdb.atoms[i].name);
		cgtop.atoms[i].resid = cgpdb.atoms[i].resid;
		sprintf(cgtop.atoms[i].resName, "%s", cgpdb.atoms[i].resName);
		cgtop.atoms[i].chain = cgpdb.atoms[i].chain;
		sprintf(cgtop.atoms[i].type, "%s", cgpdb.atoms[i].name);
		cgtop.atoms[i].charge = beads.at(i).q;
		cgtop.atoms[i].mass = beads.at(i).m;
	}
	for(b = 0; b < (int)bonds.size(); b++){
		cgtop.bonds[b].i = bonds.at(b).i;
		cgtop.bonds[b].j = bonds.at(b).j;
		cgtop.bonds[b].func = 40;
		cgtop.bonds[b].c0 = bonds.at(b).b0;
		cgtop.bonds[b].c1 = bonds.at(b).Kb;
		cgtop.bonds[b].c2 = 0.0f;
		cgtop.bonds[b].c3 = 0.0f;
	}

	for(b = 0; b < (int)natives.size(); b++){
		cgtop.pairs[b].i = natives.at(b).i;
		cgtop.pairs[b].j = natives.at(b).j;
		cgtop.pairs[b].func = 40;
		cgtop.pairs[b].c0 = natives.at(b).r0;
		cgtop.pairs[b].c1 = natives.at(b).eh;
		cgtop.pairs[b].c2 = 0.0f;
		cgtop.pairs[b].c3 = 0.0f;
	}

	for(b = 0; b < (int)exclusions.size(); b++){
		cgtop.exclusions[b].i = exclusions.at(b).i;
		cgtop.exclusions[b].j = exclusions.at(b).j;
		cgtop.exclusions[b].func = 40;
	}

	getMaskedParameter(filename, PARAMETER_TOPOLOGY);
	writeTOP(filename, &cgtop);

	printf("Files are completed.\n");

	return 0;
}

void addConnection(char* segment1, int resid1, char* bead1,
		char* segment2, int resid2, char* bead2){
	unsigned i = findBead(segment1, resid1, bead1);
	unsigned j = findBead(segment2, resid2, bead2);
	printf("Connecting: %s-%d-%s to %s-%d-%s\n",
			segment1, resid1, bead1, segment2, resid2, bead2);
	if(i < beads.size() && j < beads.size()){
		addBond(i, j, getFloatParameter(PARAMETER_KS, DEFAULT_KS), getDistance(beads.at(i), beads.at(j)));
		printf("Added: %d-%d\n", i, j);
	}

}

unsigned findBead(const char* segment, int resid, const char* bead_name){
	unsigned i = 0;
	for(i = 0; i < beads.size(); i++){
		SOPBead bead = beads.at(i);
		if(strcmp(bead.segment, segment) == 0 && bead.resid == resid && strcmp(bead.name, bead_name) == 0){
			return i;
		}
	}
	return beads.size();
}

void addConnection(SOPBead bead, Atom conn, unsigned int j){
	unsigned int l;
	printf("%s %s to %s\n", bead.resname, bead.name, conn.name);
	if(conn.name[0] == '+'){
		l = j + 1;
		while(l < beads.size() && beads.at(l).resid != bead.resid + 1){
			l ++;
		}
		while(l < beads.size() && strcmp(beads.at(l).name, &conn.name[1]) != 0 &&
				beads.at(l).resid != bead.resid + 1){
			l ++;
		}
		if(l < beads.size() && strcmp(beads.at(j).segment, beads.at(l).segment) == 0){
//			addBond(j, l, parameters::Ks.get(), getDistance(bead, beads.at(l)));
			addBond(j, l, getFloatParameter(PARAMETER_KS, DEFAULT_KS), getDistance(bead, beads.at(l)));
			printf("Added: + %d-%d\n", j, l);
		}
	} else
	if(conn.name[0] == '-'){
		l = j - 1;
		while(l > 0 && beads.at(l).resid != bead.resid - 1){
			l --;
		}
		while(l > 0 && strcmp(beads.at(l).name, &conn.name[1]) != 0){
			l --;
		}
		if(l > 0 && strcmp(beads.at(j).segment, beads.at(l).segment) == 0){
			addBond(j, l, getFloatParameter(PARAMETER_KS, DEFAULT_KS), getDistance(bead, beads.at(l)));
			printf("Added: - %d-%d\n", j, l);
		}
	} else {
		l = j + 1;
		bool added = false;
		while(l < beads.size() && bead.resid == beads.at(l).resid &&
				 strcmp(beads.at(l).name, conn.name) != 0){
			l ++;
		}
		if(l < beads.size() && strcmp(beads.at(j).segment, beads.at(l).segment) == 0 &&
				beads.at(l).resid == bead.resid){
			addBond(j, l, getFloatParameter(PARAMETER_KS, DEFAULT_KS), getDistance(bead, beads.at(l)));
			added = true;
			printf("Added: %d-%d\n", j, l);
		}
		if(!added){
			if(j - 1 >= 0){
				l = j - 1;
				while(l >= 0 && bead.resid == beads.at(l).resid &&
						strcmp(beads.at(l).name, conn.name) != 0){
					l --;
				}
				if(l >= 0 && strcmp(beads.at(j).segment, beads.at(l).segment) == 0 &&
						beads.at(l).resid == bead.resid){
					addBond(j, l, getFloatParameter(PARAMETER_KS, DEFAULT_KS), getDistance(bead, beads.at(l)));
					printf("Added: %d-%d\n", j, l);
				}
			}
		}
	}
}

void addBond(int i, int j, float Kb, float b0){
	int b;
	int found = 0;
	for(b = 0; b < (int)bonds.size(); b++){
		if((i == bonds.at(b).i && j == bonds.at(b).j) ||
				(i == bonds.at(b).j && j == bonds.at(b).i)){
			found = 1;
		}
	}
	if(i != j && found != 1){
		SOPBond bond;
		bond.i = i;
		bond.j = j;
		bond.Kb = Kb;
		bond.b0 = b0;
		bonds.push_back(bond);
	}
}

void addNative(int i, int j, float eh, float r0){
	int b;
	int found = 0;
	for(b = 0; b < (int)natives.size(); b++){
		if((i == natives.at(b).i && j == natives.at(b).j) ||
				(i == natives.at(b).j && j == natives.at(b).i)){
			found = 1;
		}
	}
	if(i != j && found != 1){
		SOPNative native;
		native.i = i;
		native.j = j;
		native.eh = eh;
		native.r0 = r0;
		natives.push_back(native);
	}
}

int checkBonded(int i, int j){
	/*int b;
	for(b = 0; b < (int)bonds.size(); b++){
		if((i == bonds.at(b).i && j == bonds.at(b).j) ||
				(i == bonds.at(b).j && j == bonds.at(b).i)){
			return 1;
		}
	}
	return 0;*/
	int b;
	for(b = 0; b < bondCount[i]; b++){
		if(bondsList[i][b] == j){
			return 1;
		}
	}
	return 0;
}

int check13Bonded(int i, int j){
	int b;
	for(b = 0; b < bond13Count[i]; b++){
		if(bonds13List[i][b] == j){
			return 1;
		}
	}
	return 0;
}

int checkNatived(int i, int j){
	int b;
	for(b = 0; b < (int)natives.size(); b++){
		if((i == natives.at(b).i && j == natives.at(b).j) ||
				(i == natives.at(b).j && j == natives.at(b).i)){
			return 1;
		}
	}
	return 0;
}

bool checkIfInXRay(int beadID){
	if(xraypdb.atomCount == -1){
		return true;
	} else {
		int i;
		for(i = 0; i < xraypdb.atomCount; i++){
			if(xraypdb.atoms[i].resid == beads.at(beadID).resid && 
				xraypdb.atoms[i].chain == beads.at(beadID).segment[0]){
					return true;
			}
		}
		printf("NOT adding contact for residue %s%d (segment '%s') that is missing in XRay structure.\n", beads.at(beadID).resname, beads.at(beadID).resid, beads.at(beadID).segment);
		return false;
	} 
}

double getDistance(SOPBead bead1, SOPBead bead2){
	double dx = bead1.x - bead2.x;
	double dy = bead1.y - bead2.y;
	double dz = bead1.z - bead2.z;
	return sqrt(dx*dx + dy*dy + dz*dz);
}

double getDistance(PDBAtom atom1, PDBAtom atom2){
	double dx = atom1.x - atom2.x;
	double dy = atom1.y - atom2.y;
	double dz = atom1.z - atom2.z;
	return sqrt(dx*dx + dy*dy + dz*dz);
}

