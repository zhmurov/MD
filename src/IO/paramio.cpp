/*
 * paramio.c
 *
 *  Created on: July 6, 2016
 *      Author: kir_min
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#define UNWRAP
#ifdef UNWRAP
# define safe_fopen fopen
# define safe_fgets fgets
# define safe_fread fread
# define DIE(format, ...) do{ printf(format, ##__VA_ARGS__); exit(-1); }while(0);
#else
# include "../Util/wrapper.h"
#endif

#include "paramio.h"

#define BUF_SIZE 256

BondCoeff readBondCoeffLineFromPARAM(FILE* paramFile);
AngleCoeff readAngleCoeffLineFromPARAM(FILE* paramFile);
GaussCoeff readGaussCoeffLineFromPARAM(FILE* paramFile);
LJ_RepulsiveCoeff readLJCoeffLineFromPARAM(FILE* paramFile);
int countRowsInPARAM(FILE* paramFile);

int readPARAM(char* filename, PARAMData* paramData){
	printf("Reading parameters from '%s'.\n", filename);
	FILE* paramFile = safe_fopen(filename, "r");
	char buffer[BUF_SIZE];
	paramData->bondCount = 0;
	paramData->angleCount = 0;
	paramData->gaussCount = 0;
	paramData->ljCount = 0;

	if (paramFile != NULL ){
		while(safe_fgets(buffer, BUF_SIZE, paramFile) != NULL){
			if(strstr(buffer, "Bond Coeffs") != 0){
				printf("Counting bonds...\n");
				paramData->bondCount = countRowsInPARAM(paramFile);
				printf("%d bonds\n", paramData->bondCount);
			}

			if(strstr(buffer, "Angle Coeffs") != 0){
				printf("Counting angles...\n");
				paramData->angleCount = countRowsInPARAM(paramFile);
				printf("%d angles\n", paramData->angleCount);
			}

			if(strstr(buffer, "Gauss Coeffs") != 0){
				printf("Counting gauss...\n");
				safe_fgets(buffer, BUF_SIZE, paramFile);//pass an empty string
				while(safe_fgets(buffer, BUF_SIZE, paramFile) != NULL && strlen(buffer) != 0 && buffer[0] != '\n' && buffer[0] != '\r'){
					if (strncmp(buffer, "#", 1) == 0){
						paramData->gaussCount++;
					}
				}
				printf("%d Gauss coeffs\n", paramData->gaussCount);
			}

			if(strstr(buffer, "LJ_Repulsive Coeffs") != 0){
				printf("Counting lj_repulsive...\n");
				safe_fgets(buffer, BUF_SIZE, paramFile);//pass an empty string
				while(safe_fgets(buffer, BUF_SIZE, paramFile) != NULL && strlen(buffer) != 0 && buffer[0] != '\n' && buffer[0] != '\r'){
					if (strncmp(buffer, "#", 1) == 0){
						paramData->ljCount++;
					}
				}
				printf("%d LJ coeffs\n", paramData->ljCount);
			}
			
		}
		paramData->bondCoeff = (BondCoeff*)calloc(paramData->bondCount, sizeof(BondCoeff));
		paramData->angleCoeff = (AngleCoeff*)calloc(paramData->angleCount, sizeof(AngleCoeff));
		paramData->gaussCoeff = (GaussCoeff*)calloc(paramData->gaussCount, sizeof(GaussCoeff));
		paramData->lj_RepulsiveCoeff = (LJ_RepulsiveCoeff*)calloc(paramData->ljCount, sizeof(LJ_RepulsiveCoeff));
		
	} else {
		DIE("ERROR: cant find parameters file '%s'.", filename);
	}
	
	rewind(paramFile);
	int count = 0;
	while(safe_fgets(buffer, BUF_SIZE, paramFile) != NULL){
		if(strstr(buffer, "Bond Coeffs") != 0){
			printf("Reading bonds coeffs.\n");
			count = 0;
			safe_fgets(buffer, BUF_SIZE, paramFile);//pass name string
			while(count < paramData->bondCount){
				paramData->bondCoeff[count] = readBondCoeffLineFromPARAM(paramFile);
				count++;
			}
			printf("Done reading bonds coeffs.\n");
		}
		if(strstr(buffer, "Angle Coeffs") != 0){
			printf("Reading angles coeffs.\n");
			count = 0;
			safe_fgets(buffer, BUF_SIZE, paramFile);//pass name string
			while(count < paramData->angleCount){
				paramData->angleCoeff[count] = readAngleCoeffLineFromPARAM(paramFile);
				count++;
			}
			printf("Done reading angles coeffs.\n");
		}
		if(strstr(buffer, "Gauss Coeffs") != 0){
			printf("Reading gauss coeffs.\n");
			count = 0;
			safe_fgets(buffer, BUF_SIZE, paramFile);//pass an empty string
			while(count < paramData->gaussCount){
				paramData->gaussCoeff[count] = readGaussCoeffLineFromPARAM(paramFile);
				count++;
			}
			printf("Done reading gauss coeffs.\n");
		}
		if(strstr(buffer, "LJ_Repulsive Coeffs") != 0){
			printf("Reading LJ coeffs.\n");
			count = 0;
			safe_fgets(buffer, BUF_SIZE, paramFile);//pass an empty string
			while(count < paramData->ljCount){
				paramData->lj_RepulsiveCoeff[count] = readLJCoeffLineFromPARAM(paramFile);
				count++;
			}
			printf("Done reading LJ coeffs.\n");
		}
	}

	fclose(paramFile);
	printf("Done reading the parameters section.\n");
	return count;

}

int countRowsInPARAM(FILE* paramFile){
	char buffer[BUF_SIZE];
	char* pch;
	int result = 0;
	int skip;
	char* eof;
	do{
		eof = safe_fgets(buffer, BUF_SIZE, paramFile);
		pch = strtok(buffer, " ");
		if(strcmp(pch, "#") != 0){
			result ++;
			skip = 0;
		} else {
			skip = 1;
		}
	} while((strcmp(pch,"0")==0 ||atoi(pch) != 0 || skip == 1) && eof != NULL);
	return result - 1;
}

BondCoeff readBondCoeffLineFromPARAM(FILE* paramFile){
	char buffer[BUF_SIZE];
	char* pch;
	BondCoeff bond;
	safe_fgets(buffer, BUF_SIZE, paramFile);

	pch = strtok(buffer, " \t");
	bond.id = atoi(pch);

	pch = strtok(NULL, " \t");
	strcpy(bond.typeName, pch);

	//printf("%s\n", bond.typeName);

	pch = strtok(NULL, " \t");
	bond.l0 = atof(pch);

	pch = strtok(NULL, " \t");
	bond.k2 = atof(pch);

	pch = strtok(NULL, " \t");
	bond.k3 = atof(pch);

	pch = strtok(NULL, " \t");
	bond.k4 = atof(pch);

	//printf("%d\t%f\t%f\t%f\t%f\n", bond.id, bond.l0, bond.k2, bond.k3, bond.k4);

	return bond;
}

AngleCoeff readAngleCoeffLineFromPARAM(FILE* paramFile){
	char buffer[BUF_SIZE];
	char* pch;
	AngleCoeff angle;
	safe_fgets(buffer, BUF_SIZE, paramFile);

	pch = strtok(buffer, " \t");
	angle.id = atoi(pch);

	pch = strtok(NULL, " \t");
	strcpy(angle.typeName, pch);

	//printf("%s\n", angle.typeName);

	pch = strtok(NULL, " \t");
	angle.theta0 = atof(pch);

	pch = strtok(NULL, " \t");
	angle.k2 = atof(pch);

	pch = strtok(NULL, " \t");
	angle.k3 = atof(pch);

	pch = strtok(NULL, " \t");
	angle.k4 = atof(pch);

	//printf("%d\t%f\t%f\t%f\t%f\n", angle.id, angle.theta0, angle.k2, angle.k3, angle.k4);

	return angle;
}

GaussCoeff readGaussCoeffLineFromPARAM(FILE* paramFile){
	char buffer[BUF_SIZE];
	char* pch;
	safe_fgets(buffer, BUF_SIZE, paramFile);
	
	if(strncmp(buffer, "#", 1) == 0){

		safe_fgets(buffer, BUF_SIZE, paramFile);//READ PARAM i, j and numberGaussians

		GaussCoeff gauss;

		pch = strtok(buffer, " \t");
		gauss.i = atoi(pch);

		pch = strtok(NULL, " \t");
		gauss.j = atoi(pch);

		pch = strtok(NULL, " \t");
		gauss.numberGaussians = atoi(pch);

		//printf("Gauss\t%d\t%d\t%d\n", gauss.i, gauss.j, gauss.numberGaussians);

		gauss.B = (float*)calloc(gauss.numberGaussians, sizeof(float));
		gauss.C = (float*)calloc(gauss.numberGaussians, sizeof(float));
		gauss.R = (float*)calloc(gauss.numberGaussians, sizeof(float));

		for (int i = 0; i < gauss.numberGaussians; ++i){
			safe_fgets(buffer, BUF_SIZE, paramFile);

			pch = strtok(buffer, " \t");
			gauss.B[i] = atof(pch);

			pch = strtok(NULL, " \t");
			gauss.C[i] = atof(pch);

			pch = strtok(NULL, " \t");
			gauss.R[i] = atof(pch);

			//printf("%f\t%f\t%f\n", gauss.B[i], gauss.C[i], gauss.R[i]);

		}

		return gauss;
	}
}

LJ_RepulsiveCoeff readLJCoeffLineFromPARAM(FILE* paramFile){
	char buffer[BUF_SIZE];
	char* pch;
	LJ_RepulsiveCoeff ljr;
	safe_fgets(buffer, BUF_SIZE, paramFile);//pass a name string

	//printf("%s\n", buffer);

	safe_fgets(buffer, BUF_SIZE, paramFile);

	if(strncmp(buffer, "#", 1) != 0){
		pch = strtok(buffer, " \t");
		ljr.i = atoi(pch);

		pch = strtok(NULL, " \t");
		ljr.j = atoi(pch);

		pch = strtok(NULL, " \t");
		ljr.A = atof(pch);

		pch = strtok(NULL, " \t");
		ljr.l = atoi(pch);

		//printf("LJ %d\t%d\t%f\t%d\n", ljr.i, ljr.j, ljr.A, ljr.L);

		return ljr;		
	}
}
