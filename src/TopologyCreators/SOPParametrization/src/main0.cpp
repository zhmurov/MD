// ITERATION = 0

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

#define FIRST_FRAME				1
#define LAST_FRAME				10001
#define STRIDE					1
#define BUF_SIZE 				256

#define T						300
#define BOLTZMANN_CONSTANT		0.00198721					// [KCal/(K*mol)]

PDB pdb;
TOPData top;
TOPData newtop;
DCD dcd;

// argv[1] - name of structure
// argv[2] - number of iterations
// argv[3] - pairType ('B' - bonds; 'P' - pairs)
// argv[4] - first frame (for dcd)
// argv[5] - last frame (for dcd)

int main(int argc, char* argv[]){
	parseParametersFile(argv[1], argc, argv);

	int iteration = atoi(argv[2]);
	printf("\n==========ITERATION %d==========\n", iteration);

	char* pairType = argv[3];
	printf("pairType = %c\n", pairType[0]);

	int first_frame, last_frame;
	((argv[4] == NULL) || (argv[5] == NULL)) ? (first_frame = FIRST_FRAME, last_frame = LAST_FRAME) :
												(first_frame = atoi(argv[4]), last_frame = atoi(argv[5]));

	printf("first_frame = %d\n", first_frame);
	printf("last_frame = %d\n", last_frame);

	char filename[BUF_SIZE];
	// coarse-grain TOP
	getMaskedParameter(filename, PARAMETER_INPUT_TOP);
	readTOP(filename, &top);
	// full-atom PDB
	getMaskedParameter(filename, PARAMETER_INPUT_FULLATOM_PDB);
	readPDB(filename, &pdb);
	// full-atom DCD
	getMaskedParameter(filename, PARAMETER_INPUT_FULLATOM_DCD);
	dcdOpenRead(&dcd, filename);
	dcdReadHeader(&dcd);

	// path output
	char path_output[BUF_SIZE];
	getMaskedParameter(path_output, PARAMETER_PATH_OUTPUT);

	int totalFrames = 0;
	int stotalFrames = 0;							// totalFrames/STRIDE
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

	// quantity of pairs for a bid
	int* Npairs = (int*)calloc(top.atomCount, sizeof(int));
	int i = 0, j = 0;
	int p = 0, b = 0, e = 0;

	int* js = (int*)calloc(top.atomCount, sizeof(int));			// js[top] = [pdb]; js[top] = [dcd]

	for(i = 0; i < top.atomCount; i++){
		for(j = 0; j < pdb.atomCount; j++){
			// resid = 1, 2, 3, ...
			if((top.atoms[i].resid == pdb.atoms[j].resid) &&
				// name = 'CA', 'CB', ...
				(strcmp(top.atoms[i].name, pdb.atoms[j].name) == 0) &&
				// segment = 'A1', 'B2', ...
				(top.atoms[i].chain == pdb.atoms[j].chain)){
				//(top.atoms[i].chain == pdb.atoms[j].segment[0])){

				js[i] = j;

				break;
			}else if(top.atoms[i].resid < pdb.atoms[j].resid){
				js[i] = -1;
			}
		}
	}

	sprintf(filename, "%sdev%d.dat", path_output, iteration);
	FILE* dev = fopen(filename, "w");

	while(dcdReadFrame(&dcd) == 0){
		totalFrames++;
		if(totalFrames >= first_frame){
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

					i = js[i];				// js[top] = [dcd]
					j = js[j];				// js[top] = [dcd]

					double x = dcd.frame.X[j] - dcd.frame.X[i];	// [angstr]
					double y = dcd.frame.Y[j] - dcd.frame.Y[i];	// [angstr]
					double z = dcd.frame.Z[j] - dcd.frame.Z[i];	// [angsrt]

					double d = sqrt(x*x + y*y + z*z);
					mean_dev[p] += d;
					disp[p] += d*d;

					fprintf(dev, "%f\t", d);
				}
				fprintf(dev, "\n");
			}
		}
		if(totalFrames == last_frame){
			break;
		}
	}

	stotalFrames = float(totalFrames+1 - first_frame)/float(STRIDE);
	printf("totalFrames = %d\n", totalFrames);
	printf("Number of frames used for data collection = %d\n", stotalFrames);

	for(p = 0; p < pairsCount; p++){
		mean_dev[p] = mean_dev[p]/stotalFrames;						// [angstr]
		disp[p] = disp[p]/stotalFrames - mean_dev[p]*mean_dev[p];	// [angstr^2]
	}

	for(i = 0; i < top.atomCount; i++){
		Npairs[i] /= stotalFrames;
	}

	//write file
	sprintf(filename, "%smeandev_disp%d.dat", path_output, iteration);
	FILE* meandev_disp_fwrite = fopen(filename, "w");
	sprintf(filename, "%seh%d.dat", path_output, iteration);
	FILE* eh_fwrite = fopen(filename, "w"); // eh is the strength of the non-bonded interactions
	sprintf(filename, "%seh_average.dat", path_output);
	FILE* eha_fwrite = fopen(filename, "w");

	sprintf(filename, "%salpha.dat", path_output);
	FILE* alpha_fwrite = fopen(filename, "w");

	double* eh = (double*)calloc(pairsCount, sizeof(double));
	double eh_aver = 0.0;

	for(p = 0; p < pairsCount; p++){

		fprintf(meandev_disp_fwrite, "%f\t", mean_dev[p]);		// [angstr]
		fprintf(meandev_disp_fwrite, "%f\n", disp[p]);			// [angstr]^2

		float N;

		if(pairType[0] == 'P'){
			i = top.pairs[p].i;
			j = top.pairs[p].j;

			N = (Npairs[i] + Npairs[j])/2.0f;
			//top.pairs.c0 is the equilibrium distance taken from the initial structure
			//eh[p] = BOLTZMANN_CONSTANT*T*top.pairs[p].c0*top.pairs[p].c0/(72.0f*disp[p]*N);
			eh[p] = BOLTZMANN_CONSTANT*T*mean_dev[p]*mean_dev[p]/(72.0f*disp[p]*N);
		}
		else if(pairType[0] == 'B'){
			i = top.bonds[p].i;
			j = top.bonds[p].j;

			N = (Npairs[i] + Npairs[j])/2.0f;
			if(top.bonds[p].c1 != 8370.0f) eh[p] = 72.0f*6.276f/(mean_dev[p]*mean_dev[p]/100.0f);
			else eh[p] = top.bonds[p].c1;
		}
		eh_aver += eh[p];
		fprintf(eh_fwrite, "%3d\t%3d\t%f\n", i, j, eh[p]);
	}
	fprintf(eha_fwrite, "iteration%4d:\t%f\n", iteration, eh_aver/float(pairsCount));

	fclose(meandev_disp_fwrite);
	fclose(eh_fwrite);
	fclose(eha_fwrite);
	fclose(alpha_fwrite);
	fclose(dev);

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
			newtop.bonds[b].c0 = mean_dev[b];					// [angstr]
			newtop.bonds[b].c1 = eh[b];							// [kJ/(mol*nm^2)]
		}
	}
	for(p = 0; p < newtop.pairsCount; p++){
		// pairType[0] == 'P'
		newtop.pairs[p].i = top.pairs[p].i;
		newtop.pairs[p].j = top.pairs[p].j;
		newtop.pairs[p].func = top.pairs[p].func;
		newtop.pairs[p].c0 = mean_dev[p];						// [angstr]
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
	free(Npairs);
	free(js);
	free(eh);
	// TODO
	free(newtop.atoms);
	free(newtop.bonds);
	free(newtop.pairs);
	free(newtop.exclusions);

	printf("==========ITERATION 0 SUCCESS ==========\n");
	return 0;
}
