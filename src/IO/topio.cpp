/*
 * topio.c
 *
 *  Created on: May 24, 2009
 *      Author: zhmurov
 *  Changes: 16.08.2016
 *	Author: kir_min
 *  Changes: 28.03.2017
 *  	Author: ilya_kir
 *  	Added function getIndexInTOP
 *  Changes: 03.04.2017
 *  	Author: ilya_kir
 *  	Added reading func to function readExclusionLineFromTOP 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef UNWRAP
# define safe_fopen fopen
# define safe_fgets fgets
# define safe_fread fread
# define DIE(format, ...) do{ printf(format, ##__VA_ARGS__); exit(-1); }while(0);
#else
# include "../Util/wrapper.h"
#endif

#include "topio.h"

#define BUF_SIZE 256

int countRowsInTOP(FILE* topFile);
TOPAtom readAtomLineFromTOP(FILE* topFile);
TOPPair readPairLineFromTOP(FILE* topFile);
TOPAngle readAngleLineFromTOP(FILE* topFile);
TOPDihedral readDihedralLineFromTOP(FILE* topFile);
TOPExclusion readExclusionLineFromTOP(FILE* topFile);
void savePair(FILE* topFile, TOPPair pair);

//Added 28.03.17
int getIndexInTOP(int nr, TOPData* topData){
	if(topData->ids[nr] != -1){
		return topData->ids[nr];
	} else {
		printf("Atom with index %d not found in the topology file.\n", nr);
		exit(0);
	}
}

int readTOP(const char* filename, TOPData* topData){
	printf("Reading topology from '%s'.\n", filename);
	FILE* topFile = safe_fopen(filename, "r");
	char buffer[BUF_SIZE];
	if (topFile != NULL ){
		while(safe_fgets(buffer, BUF_SIZE, topFile) != NULL){
			if(strstr(buffer, "[ atoms ]") != 0){
				printf("Counting atoms...\n");
				topData->atomCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->atomCount);
			}
			if(strstr(buffer, "[ bonds ]") != 0){
				printf("Counting bonds...\n");
				topData->bondCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->bondCount);
			}
			if(strstr(buffer, "[ pairs ]") != 0){
				printf("Counting pairs...\n");
				topData->pairsCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->pairsCount);
			}
			if(strstr(buffer, "[ angles ]") != 0){
				printf("Counting angles...\n");
				topData->angleCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->angleCount);
			}
			if(strstr(buffer, "[ dihedrals ]") != 0){
				printf("Counting dihedrals...\n");
				topData->dihedralCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->dihedralCount);
			}
			if(strstr(buffer, "[ exclusions ]") != 0){
				printf("Counting exclusions...\n");
				topData->exclusionCount = countRowsInTOP(topFile);
				printf("%d found.\n", topData->exclusionCount);
			}
		}
		topData->atoms = (TOPAtom*)calloc(topData->atomCount, sizeof(TOPAtom));
		topData->bonds = (TOPPair*)calloc(topData->bondCount, sizeof(TOPPair));
		topData->pairs = (TOPPair*)calloc(topData->pairsCount, sizeof(TOPPair));
		topData->angles = (TOPAngle*)calloc(topData->angleCount, sizeof(TOPAngle));
		topData->dihedrals = (TOPDihedral*)calloc(topData->dihedralCount, sizeof(TOPDihedral));
		topData->exclusions = (TOPExclusion*)calloc(topData->exclusionCount, sizeof(TOPExclusion));
	} else {
		DIE("ERROR: cant find topology file '%s'.", filename);
	}

	rewind(topFile);

	int count = 0;
	while(safe_fgets(buffer, BUF_SIZE, topFile) != NULL){
		if(strstr(buffer, "[ atoms ]") != 0){
			count = 0;
			while(count < topData->atomCount){
				topData->atoms[count] = readAtomLineFromTOP(topFile);
				if(topData->atoms[count].id != -1){
					count++;
				}
			}
		}
		if(strstr(buffer, "[ bonds ]") != 0){
			count = 0;
			while(count < topData->bondCount){
				topData->bonds[count] = readPairLineFromTOP(topFile);
				if(topData->bonds[count].i != -1){
					count++;
				}
			}
		}
		if(strstr(buffer, "[ pairs ]") != 0){
			count = 0;
			while(count < topData->pairsCount){
				topData->pairs[count] = readPairLineFromTOP(topFile);
				if(topData->pairs[count].i != -1){
					count++;
				}
			}
		}
		if(strstr(buffer, "[ angles ]") != 0){
			printf("Reading angles...\n");
			count = 0;
			while(count < topData->angleCount){
				topData->angles[count] = readAngleLineFromTOP(topFile);
				if(topData->angles[count].i != -1){
					count++;
				}
			}
		}
		if(strstr(buffer, "[ dihedrals ]") != 0){
			printf("Reading dihedrals...\n");
			count = 0;
			while(count < topData->dihedralCount){
				topData->dihedrals[count] = readDihedralLineFromTOP(topFile);
				if(topData->dihedrals[count].i != -1){
					count++;
				}
			}
		}
		if(strstr(buffer, "[ exclusions ]") != 0){
			printf("Reading exclusions...\n");
			count = 0;
			while(count < topData->exclusionCount){
				topData->exclusions[count] = readExclusionLineFromTOP(topFile);
				if(topData->exclusions[count].i != -1){
					count++;
				}
			}
		}
	}

	//Added 28.03.17
	int maxnr = 0;
	for(int i = 0; i < topData->atomCount; i++){
		if(topData->atoms[i].id > maxnr){
			maxnr = topData->atoms[i].id;
		}
	}
	topData->ids = (int*)calloc((maxnr+1) , sizeof(int));
	for(int i = 0; i <= maxnr; i++){
		topData->ids[i] = -1;
	}
	for(int i = 0; i < topData->atomCount; i++){
		topData->ids[topData->atoms[i].id] = i;
	}

	fclose(topFile);
	printf("Done reading the topology section.\n");
	return count;
}

void writeTOP(const char* filename, TOPData* topData){
	int i;
	FILE* topFile = safe_fopen(filename, "w");
	fprintf(topFile, "; Created by topio.c utility\n\n");
	fprintf(topFile, "[ atoms ]\n");
	fprintf(topFile, ";   nr       type  resnr residue  atom   cgnr     charge       mass\n");
	for(i = 0; i < topData->atomCount; i++){
		fprintf(topFile, "%6d", i);
		fprintf(topFile, "%11s", topData->atoms[i].type);
		fprintf(topFile, "%7d", topData->atoms[i].resid);
		fprintf(topFile, "%7s", topData->atoms[i].resName);
		fprintf(topFile, "%7s", topData->atoms[i].name);
		fprintf(topFile, "%7c", topData->atoms[i].chain);
		fprintf(topFile, "%11.2f", topData->atoms[i].charge);
		fprintf(topFile, "%11.3f", topData->atoms[i].mass);
		fprintf(topFile, "\n");
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ bonds ]\n");
	fprintf(topFile, ";  ai    aj funct            c0            c1            c2            c3\n");
	for(i = 0; i < topData->bondCount; i++){
		savePair(topFile, topData->bonds[i]);
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ pairs ]\n");
	fprintf(topFile, ";  ai    aj funct            c0            c1            c2            c3\n");
	for(i = 0; i < topData->pairsCount; i++){
		savePair(topFile, topData->pairs[i]);
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ exclusions ]\n");
	fprintf(topFile, ";  ai    aj funct\n");
	for(i = 0; i < topData->exclusionCount; i++){
		fprintf(topFile, "%5d", topData->exclusions[i].i);
		fprintf(topFile, " ");
		fprintf(topFile, "%5d", topData->exclusions[i].j);
		fprintf(topFile, " ");
		fprintf(topFile, "%5d", topData->exclusions[i].func);
		fprintf(topFile, "\n");
	}

	fclose(topFile);

}

void savePair(FILE* topFile, TOPPair pair){
	fprintf(topFile, "%5d", pair.i);
	fprintf(topFile, " ");
	fprintf(topFile, "%5d", pair.j);
	fprintf(topFile, " ");
	fprintf(topFile, "%5d", pair.func);
	if(pair.c0 != 0){
		fprintf(topFile, " ");
		fprintf(topFile, "%10.5f", pair.c0);
		if(pair.c1 != 0){
			fprintf(topFile, " ");
			fprintf(topFile, "%10.5f", pair.c1);
			if(pair.c2 != 0){
				fprintf(topFile, " ");
				fprintf(topFile, "%10.5f", pair.c2);
				if(pair.c3 != 0){
					fprintf(topFile, " ");
					fprintf(topFile, "%10.5f", pair.c3);
				}
			}
		}
	}
	fprintf(topFile, "\n");
}


/*void saveTOP(char* filename){
	int i;
	FILE* topFile = safe_fopen(filename, "w");
	fprintf(topFile, "; Created by topio.c utility\n\n");
	fprintf(topFile, "[ atoms ]\n");
	fprintf(topFile, ";   nr       type  resnr residue  atom   cgnr     charge       mass\n");
	for(i = 0; i < sop.aminoCount; i++){
		fprintf(topFile, "%6d", i);
		fprintf(topFile, "%11s", sop.aminos[i].name);
		fprintf(topFile, "%7d", sop.aminos[i].resid);
		fprintf(topFile, "%7s", sop.aminos[i].resName);
		fprintf(topFile, "%7s", sop.aminos[i].name);
		fprintf(topFile, "%7c", sop.aminos[i].chain);
		fprintf(topFile, "%11.2f", sop.aminos[i].occupancy);
		fprintf(topFile, "%11.3f", sop.aminos[i].beta);
		fprintf(topFile, "\n");
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ bonds ]\n");
	fprintf(topFile, ";  ai    aj funct            c0            c1            c2            c3\n");
	for(i = 0; i < sop.bondCount; i++){
		TOPPair pair;
		pair.i = sop.bonds[i].i;
		pair.j = sop.bonds[i].j;
		pair.func = 1;
		pair.c0 = sop.bonds[i].r0;
		pair.c1 = 0.0f;
		savePair(topFile, pair);
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ native ]\n");
	fprintf(topFile, ";  ai    aj funct            c0            c1            c2            c3\n");
	for(i = 0; i < sop.nativeCount; i++){
		TOPPair pair;
		pair.i = sop.natives[i].i;
		pair.j = sop.natives[i].j;
		pair.func = 1;
		pair.c0 = sop.natives[i].r0;
		pair.c1 = sop.natives[i].eh;
		pair.c2 = 0.0f;
		savePair(topFile, pair);
	}

	fprintf(topFile, "\n");

	fprintf(topFile, "[ pairs ]\n");
	fprintf(topFile, ";  ai    aj funct            c0            c1            c2            c3\n");
	for(i = 0; i < sop.pairCount; i++){
		TOPPair pair;
		pair.i = sop.pairs[i].i;
		pair.j = sop.pairs[i].j;
		pair.func = 1;
		pair.c0 = 0.0f;
		savePair(topFile, pair);
	}


	fclose(topFile);

}*/


/*void savePair(FILE* topFile, TOPPair pair){
	fprintf(topFile, "%5d", pair.i);
	fprintf(topFile, " ");
	fprintf(topFile, "%5d", pair.j);
	fprintf(topFile, " ");
	fprintf(topFile, "%5d", pair.func);
	if(pair.c0 != 0){
		fprintf(topFile, " ");
		fprintf(topFile, "%10.5f", pair.c0);
		if(pair.c1 != 0){
			fprintf(topFile, " ");
			fprintf(topFile, "%10.5f", pair.c1);
			if(pair.c2 != 0){
				fprintf(topFile, " ");
				fprintf(topFile, "%10.5f", pair.c2);
				if(pair.c3 != 0){
					fprintf(topFile, " ");
					fprintf(topFile, "%10.5f", pair.c3);
				}
			}
		}
	}
	fprintf(topFile, "\n");
}*/


int countRowsInTOP(FILE* topFile){
	char buffer[BUF_SIZE];
	char* pch;
	int result = 0;
	int skip;
	char* eof;
	do{
		eof = safe_fgets(buffer, BUF_SIZE, topFile);
		pch = strtok(buffer, " \t");
		//printf("'%s'\n", pch);
		if(strcmp(pch, ";") != 0){
			result ++;
			skip = 0;
		} else {
			skip = 1;
		}
	} while((strcmp(pch,"0")==0 ||atoi(pch) != 0 || skip == 1) && eof != NULL);
	return result - 1;
}

TOPAtom readAtomLineFromTOP(FILE* topFile){
	char buffer[BUF_SIZE];
	char* pch;
	TOPAtom atom;
	safe_fgets(buffer, BUF_SIZE, topFile);

	if(strncmp(buffer, ";", 1) != 0){
		pch = strtok(buffer, " \t");
		atom.id = atoi(pch);

		pch = strtok(NULL, " \t");
		strcpy(atom.type, pch);

		pch = strtok(NULL, " \t");
		atom.resid = atoi(pch);

		pch = strtok(NULL, " \t");
		strcpy(atom.resName, pch);

		pch = strtok(NULL, " \t");
		strcpy(atom.name, pch);

		pch = strtok(NULL, " \t");
		atom.chain = pch[0];

		pch = strtok(NULL, " \t");
		atom.charge = atof(pch);

		pch = strtok(NULL, " \t");
		atom.mass = atof(pch);
	} else {
		atom.id = -1;
	}

	return atom;
}

TOPPair readPairLineFromTOP(FILE* topFile){

	TOPPair pair;
	pair.c0 = 0.0f;
	pair.c1 = 0.0f;
	pair.c2 = 0.0f;
	pair.c3 = 0.0f;
	char buffer[BUF_SIZE];
	char* pch;
	safe_fgets(buffer, BUF_SIZE, topFile);
	if(strncmp(buffer, ";", 1) != 0){
		pch = strtok(buffer, " \t");
		pair.i = atoi(pch);

		pch = strtok(NULL, " \t");
		pair.j = atoi(pch);

		pch = strtok(NULL, " \t");
		pair.func = atoi(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return pair;
		}
		pair.c0 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return pair;
		}
		pair.c1 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return pair;
		}
		pair.c2 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return pair;
		}
		pair.c3 = atof(pch);
	} else {
		pair.i = -1;
	}
	return pair;
}

TOPAngle readAngleLineFromTOP(FILE* topFile){
	TOPAngle angle;
	angle.c0 = 0.0;
	angle.c1 = 0.0;
	angle.c2 = 0.0;
	angle.c3 = 0.0;
	char buffer[BUF_SIZE];
	char* pch;
	safe_fgets(buffer, BUF_SIZE, topFile);
	if(strncmp(buffer, ";", 1) != 0){
		pch = strtok(buffer, " \t");
		angle.i = atoi(pch);

		pch = strtok(NULL, " \t");
		angle.j = atoi(pch);

		pch = strtok(NULL, " \t");
		angle.k = atoi(pch);

		pch = strtok(NULL, " \t");
		angle.func = atoi(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return angle;
		}
		angle.c0 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return angle;
		}
		angle.c1 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return angle;
		}
		angle.c2 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return angle;
		}
		angle.c3 = atof(pch);
	} else {
		angle.i = -1;
	}
	return angle;
}

TOPDihedral readDihedralLineFromTOP(FILE* topFile){
	TOPDihedral dihedral;
	dihedral.c0 = 0.0;
	dihedral.c1 = 0.0;
	dihedral.c2 = 0.0;
	dihedral.c3 = 0.0;
	char buffer[BUF_SIZE];
	char* pch;
	safe_fgets(buffer, BUF_SIZE, topFile);
	if(strncmp(buffer, ";", 1) != 0){
		pch = strtok(buffer, " \t");
		dihedral.i = atoi(pch);

		pch = strtok(NULL, " \t");
		dihedral.j = atoi(pch);

		pch = strtok(NULL, " \t");
		dihedral.k = atoi(pch);

		pch = strtok(NULL, " \t");
		dihedral.l = atoi(pch);

		pch = strtok(NULL, " \t");
		dihedral.func = atoi(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return dihedral;
		}
		dihedral.c0 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return dihedral;
		}
		dihedral.c1 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return dihedral;
		}
		dihedral.c2 = atof(pch);

		pch = strtok(NULL, " \t");
		if(pch == NULL){
			return dihedral;
		}
		dihedral.c3 = atof(pch);
	} else {
		dihedral.i = -1;
	}
	return dihedral;
}

TOPExclusion readExclusionLineFromTOP(FILE* topFile){
	char buffer[BUF_SIZE];
	char* pch;
	TOPExclusion exclusion;
	safe_fgets(buffer, BUF_SIZE, topFile);

	if(strncmp(buffer, ";", 1) != 0){
		pch = strtok(buffer, " \t");
		exclusion.i = atoi(pch);

		pch = strtok(NULL, " \t");
		exclusion.j = atoi(pch);

	//Added 03.04.17
		pch = strtok(NULL, " \t");
		exclusion.func = atoi(pch);
	} else {
		exclusion.i = -1;
	}
	return exclusion;
}

