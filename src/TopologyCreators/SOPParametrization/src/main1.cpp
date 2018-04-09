// ITERATION != 0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <cmath>

#include "../../../parameters.h"
#include "../../../IO/configreader.h"

#include "../../../IO/pdbio.h"
#include "../../../IO/topio.h"
#include "../../../IO/dcdio.h"

#define STRIDE			10
#define BUF_SIZE		256

#define ALPHA			0.1
#define T				300
#define BOLTZMANN_CONSTANT	0.00198721					// [KCal/(K*mol)]

TOPData top;
TOPData	newtop;
DCD dcd;

int main(int argc, char* argv[]){
	parseParametersFile(argv[1], argc, argv);

	int iteration = atoi(argv[2]);
	printf("==========ITERATION %d==========\n", iteration);

	char* pairType = argv[3];
	printf("pairType = %c\n", pairType[0]);

	char filename[BUF_SIZE];
	// coarse-grain TOP
	getMaskedParameter(filename, PARAMETER_INPUT_TOP);
	readTOP(filename, &top);
	// coarse-grain DCD
	getMaskedParameter(filename, PARAMETER_INPUT_DCD);
	dcdOpenRead(&dcd, filename);
	dcdReadHeader(&dcd);

	// path output
	char path_output[BUF_SIZE];
	getMaskedParameter(path_output, PARAMETER_PATH_OUTPUT);

	int totalFrames = 0;
	int stotalFrames = 0;		// totalFrames/STRIDE
	int dcdCount = dcd.header.N;
	dcd.frame.X = (float*)calloc(dcdCount, sizeof(float));
	dcd.frame.Y = (float*)calloc(dcdCount, sizeof(float));
	dcd.frame.Z = (float*)calloc(dcdCount, sizeof(float));

	int pairsCount = 0;								// total number of pairs
	if(pairType[0] == 'P'){
		pairsCount = top.pairsCount;
	}
	else if(pairType[0] == 'B'){
		pairsCount = top.bondCount;
	}
	else{
		printf("ERROR: pairType VALUE (argv[3]) IS WRONG");
		exit(0);
	}

	// mean deviation
	double* mean_dev = (double*)calloc(pairsCount, sizeof(double));
	// dispersion (standard deviation squared)
	double* disp = (double*)calloc(pairsCount, sizeof(double));
	// dispersion (another formula)
	double* disp2 = (double*)calloc(pairsCount, sizeof(double));

	// quantity of pairs for a bid
	int* Npairs = (int*)calloc(top.atomCount, sizeof(int));
	int i = 0, j = 0;
	int p = 0, b = 0, e = 0;

	sprintf(filename, "%sdev%d.dat", path_output, iteration);
	FILE* dev = fopen(filename, "w");

	while(dcdReadFrame(&dcd) == 0){
		totalFrames++;
		if((totalFrames % STRIDE) == 0){
			for(p = 0; p < pairsCount; p++){

				if(pairType[0] == 'P'){
					i = top.pairs[p].i;
					j = top.pairs[p].j;
				}
				else if(pairType[0] == 'B'){
					i = top.bonds[p].i;
					j = top.bonds[p].j;
				}

				Npairs[i]++;
				Npairs[j]++;

				double x = (dcd.frame.X[j] - dcd.frame.X[i]);	// [angstr]
				double y = (dcd.frame.Y[j] - dcd.frame.Y[i]);	// [angstr]
				double z = (dcd.frame.Z[j] - dcd.frame.Z[i]);	// [angsrt]

				double d = sqrt(x*x + y*y + z*z);
				mean_dev[p] += d;
				disp[p] += d*d;
				if(pairType[0] == 'P') disp2[p] += (d - top.pairs[p].c0)*(d - top.pairs[p].c0);
				else if(pairType[0] == 'B') disp2[p] += (d - top.bonds[p].c0)*(d - top.bonds[p].c0);

				fprintf(dev, "%f\t", d);
			}
			fprintf(dev, "\n");
		}
	}

	stotalFrames = totalFrames/STRIDE;
	printf("totalFrames = %d\n", totalFrames);
	printf("totalFrames/STRIDE = %d\n", stotalFrames);

	for(p = 0; p < pairsCount; p++){
		mean_dev[p] = mean_dev[p]/stotalFrames;							// [angstr]
		disp[p] = disp[p]/stotalFrames - mean_dev[p]*mean_dev[p];		// [angstr^2]
		disp2[p] = disp2[p]/stotalFrames;								// [angstr^2]
	}
	for(i = 0; i < top.atomCount; i++){
		Npairs[i] /= stotalFrames;
	}

	//read file
	sprintf(filename, "%smeandev_disp0.dat", path_output);
	FILE* meandev_disp_fread = fopen(filename, "r");
	sprintf(filename, "%seh%d.dat", path_output, (iteration-1));
	FILE* eh_fread = fopen(filename, "r");

	//write file
	sprintf(filename, "%smeandev_disp%d.dat", path_output, iteration);
	FILE* meandev_disp_fwrite = fopen(filename, "w");
	sprintf(filename, "%seh%d.dat", path_output, iteration);
	FILE* eh_fwrite = fopen(filename, "w");
	sprintf(filename, "%seh_average.dat", path_output);
	FILE* eha_fwrite = fopen(filename, "a");

	sprintf(filename, "%salpha.dat", path_output);
	FILE* alpha_fwrite = fopen(filename, "a");

	double* eh = (double*)calloc(pairsCount, sizeof(double));
	double* eh_file = (double*)calloc(pairsCount, sizeof(double));		// eh from file
	double* disp_file = (double*)calloc(pairsCount, sizeof(double));	// disp from file
	double eh_aver = 0.0;

	char stringf[BUF_SIZE];	// string from file
	char* pch;

	double alpha = 0.0;

	for (p = 0; p < pairsCount; p++){

		if(pairType[0] == 'P'){
			i = top.pairs[p].i;
			j = top.pairs[p].j;
		}
		else if(pairType[0] == 'B'){
			i = top.bonds[p].i;
			j = top.bonds[p].j;
		}

		float N = (Npairs[i] + Npairs[j])/2.0f;

		fprintf(meandev_disp_fwrite, "%f\t", mean_dev[p]);		// [angstr]
		fprintf(meandev_disp_fwrite, "%f\t", disp[p]);			// [angstr^2]
		fprintf(meandev_disp_fwrite, "%f\n", disp2[p]);			// [angstr^2]

		if(fgets(stringf, BUF_SIZE, eh_fread) != NULL){
			pch = strtok(stringf, " \t");
			int i_eh_file = atoi(pch);

			pch = strtok(NULL, " \t");
			int j_eh_file = atoi(pch);

			if ((i != i_eh_file) || (j != j_eh_file)){
				printf("WARNING;\titeration = %d;\tp = %d\ti = %d; j = %d; i_eh_file = %d; j_eh_file = %d\n", iteration, p, i, j, i_eh_file, j_eh_file);
			}

			pch = strtok(NULL, " \t");
			eh_file[p] = atof(pch);
		}else{
			printf("ERROR: eh file not found!\n");
			exit(0);
		}

		if(fgets(stringf, BUF_SIZE, meandev_disp_fread) != NULL){
			pch = strtok(stringf, " \t");
			pch = strtok(NULL, " \t");
			disp_file[p] = atof(pch);
			pch = strtok(NULL, " \t");
		}else{
			printf("ERROR: dev file not found!\n");
			exit(0);
		}


		if(pairType[0] == 'P'){
			alpha = ALPHA;
			while(fabs(alpha*BOLTZMANN_CONSTANT*T*top.pairs[p].c0*top.pairs[p].c0*(1.0f/disp_file[p] - 1.0f/disp[p])/(72.0f*N)) > 0.05){
				alpha /= 2.0;
				fprintf(alpha_fwrite, "ITER %5d\t p = %d\t i,j = %d,%d\tDecreasing alpha to %f\n", iteration, p, i, j, alpha);
			}
			eh[p] = eh_file[p] + alpha*BOLTZMANN_CONSTANT*T*top.pairs[p].c0*top.pairs[p].c0*(1.0f/disp_file[p] - 1.0f/disp[p])/(72.0f*N);		// // [KCal/mol]
		}
		else if(pairType[0] == 'B'){
			//if(top.bonds[p].c1 != 8370.0f) eh[p] = eh_file[p] + ALPHA*BOLTZMANN_CONSTANT*T*(1.0f/disp_file[p] - 1.0f/disp[p]);
			if(top.bonds[p].c1 != 8370.0f) {
					alpha = ALPHA;
					while(fabs(eh_file[p]*alpha*(disp[p]/100.0 - disp_file[p]/100.0)) > 0.05){
						alpha /= 2.0;
						fprintf(alpha_fwrite, "ITER %5d\t p = %d\t i,j = %d,%d\tDecreasing alpha to %f\n", iteration, p, i, j, alpha);
					}
					eh[p] = eh_file[p]/(1.0f - eh_file[p]*alpha*(disp[p]/100.0 - disp_file[p]/100.0)); 	// [kJ/(mol*nm^2)]
			}
			else eh[p] = top.bonds[p].c1;																// [kJ/(mol*nm^2)]
		}

		if(eh[p] < 0.0f){
			eh[p] = 0.0f;
		}

		eh_aver += eh[p];
		fprintf(eh_fwrite, "%3d\t%3d\t%f\n", i, j, eh[p]);
	}
	fprintf(eha_fwrite, "iteration%4d:\t%f\n", iteration, eh_aver/pairsCount);

	fclose(meandev_disp_fread);
	fclose(meandev_disp_fwrite);
	fclose(eh_fread);
	fclose(eh_fwrite);
	fclose(eha_fwrite);
	fclose(alpha_fwrite);

	// newTOP
	newtop.atoms = (TOPAtom*)calloc(top.atomCount, sizeof(TOPAtom));
	newtop.bonds = (TOPPair*)calloc(top.bondCount, sizeof(TOPPair));
	newtop.pairs = (TOPPair*)calloc(top.pairsCount, sizeof(TOPPair));
	newtop.exclusions = (TOPExclusion*)calloc(top.exclusionCount, sizeof(TOPExclusion));
	newtop.atomCount = top.atomCount;
	newtop.bondCount = top.bondCount;
	newtop.pairsCount = top.pairsCount;
	newtop.exclusionCount = top.exclusionCount;

	for(i = 0; i < newtop.atomCount; i++){
		newtop.atoms[i].id = top.atoms[i].id;
		strcpy(newtop.atoms[i].type, top.atoms[i].type);
		newtop.atoms[i].resid = top.atoms[i].resid;
		strcpy(newtop.atoms[i].resName, top.atoms[i].resName);
		strcpy(newtop.atoms[i].name, top.atoms[i].name);
		newtop.atoms[i].chain = top.atoms[i].chain;
		newtop.atoms[i].charge = top.atoms[i].charge;
		newtop.atoms[i].mass = top.atoms[i].mass;
	}
	for(b = 0; b < newtop.bondCount; b++){
		newtop.bonds[b].i = top.bonds[b].i;
		newtop.bonds[b].j = top.bonds[b].j;
		newtop.bonds[b].func = top.bonds[b].func;
		// pairType[0] == 'P'
		newtop.bonds[b].c0 = top.bonds[b].c0;					// [angstr]
		newtop.bonds[b].c1 = top.bonds[b].c1;					// [kJ/(mol*nm^2)]
		if(pairType[0] == 'B'){
			newtop.bonds[b].c1 = eh[b];							// [kJ/(mol*nm^2)]
		}
	}
	for(p = 0; p < newtop.pairsCount; p++){
		newtop.pairs[p].i = top.pairs[p].i;
		newtop.pairs[p].j = top.pairs[p].j;
		newtop.pairs[p].func = top.pairs[p].func;
		newtop.pairs[p].c0 = top.pairs[p].c0;					// [angstr]
		newtop.pairs[p].c1 = eh[p];								// [KCal/mol]
	}
	for(e = 0; e < newtop.exclusionCount; e++){
		newtop.exclusions[e].i = top.exclusions[e].i;
		newtop.exclusions[e].j = top.exclusions[e].j;
		newtop.exclusions[e].func = top.exclusions[e].func;
	}
	// coarse-grain newTOP
	char newtop_filename[BUF_SIZE];
	getMaskedParameter(newtop_filename, PARAMETER_OUTPUT_NEWTOP_FILENAME);
	sprintf(filename, "%s%d.top", newtop_filename, iteration);
	writeTOP(filename, &newtop);

	// cleaning memory
	dcdClose(dcd);
	free(mean_dev);
	free(disp);
	free(disp2);
	free(Npairs);
	free(eh);
	free(eh_file);
	free(disp_file);
	// TODO
	free(newtop.atoms);
	free(newtop.bonds);
	free(newtop.pairs);
	free(newtop.exclusions);

	printf("==========ITERATION %d SUCCESS ==========\n", iteration);
	return 0;
}
