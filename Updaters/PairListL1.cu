/*
 * PairListL1.cu
 *
 *  Created on: 29.08.2012
 *      Author: zhmurov
 */

#include "PairListL1.cuh"
//#include "ComputationalArrays.h"

PairListL1::PairListL1(MDData *mdd, std::vector<int2> exclusions, float cutoff, float cuton, int frequence) : PairlistUpdater(mdd)
{
	this->mdd = mdd;
	this->frequence = frequence;
//	blockCount = (mdd->N)/DEFAULT_BLOCK_SIZE + 1;
//	blockSize = DEFAULT_BLOCK_SIZE;
	this->cutoff = cutoff;//getFloatParameter(PARAMETER_POSSIBLE_PAIRLIST_CUTOFF);
	cutoffSq = cutoff*cutoff;
	//float nbCutoff = getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	float diff = 0.5f*(cutoff - cuton);
	drsq_up = diff*diff;
	//getIntegerParameter(PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE);

	h_excl.count = (int*)calloc(mdd->N, sizeof(int));

	cudaMalloc((void**)&d_excl.count, mdd->N*sizeof(int));
	/*std::vector<int> exclTypes;
	ComputationalArrays ca(&top, &exclTypes);

	if(hasParameter(PARAMETER_EXCLUDE_BOND_TYPES)){
		exclTypes = getIntegerArrayParameter(PARAMETER_EXCLUDE_BOND_TYPES);
	}*/

	int i, j;
	//std::vector<int2> exclusions;
	//ca.GetExclusionList(&exclusions);

	for(i = 0; i < mdd->N; i++){
		h_excl.count[i] = 0;
	}
	for(i = 0; i < exclusions.size(); i++){
		h_excl.count[exclusions.at(i).x] ++;
		h_excl.count[exclusions.at(i).y] ++;
	}

	int maxExclPerAtom = 0;
	for(i = 0; i < mdd->N; i++){
		if(maxExclPerAtom < h_excl.count[i]){
			maxExclPerAtom = h_excl.count[i];
		}
	}
	maxExclPerAtom++;

	h_excl.list = (int*)calloc(mdd->widthTot*maxExclPerAtom, sizeof(int));
	cudaMalloc((void**)&d_excl.list, mdd->widthTot*maxExclPerAtom*sizeof(int));

	for(i = 0; i < mdd->N; i++){
		h_excl.count[i] = 0;
		for(j = 0; j < maxExclPerAtom; j++){
			h_excl.list[h_excl.count[j]*mdd->widthTot + i] = -1;
		}
	}

	for(i = 0; i < exclusions.size(); i++){
		int a1 = exclusions.at(i).x;
		int a2 = exclusions.at(i).y;
		h_excl.list[h_excl.count[a1]*mdd->widthTot + a1] = a2;
		h_excl.list[h_excl.count[a2]*mdd->widthTot + a2] = a1;
		h_excl.count[a1] ++;
		h_excl.count[a2] ++;
	}

	/*for(i = 0; i < mdd->N; i++){
		printf("%d (%d): ", i, h_excl.count[i]);
		for(j = 0; j < h_excl.count[i]; j++){
			printf("%d  ", h_excl.list[j*mdd->widthTot + i]);
		}
		printf("\n");
	}*/

	cudaMemcpy(d_excl.count, h_excl.count, mdd->N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_excl.list, h_excl.list, mdd->N*maxExclPerAtom*sizeof(int), cudaMemcpyHostToDevice);

//	h_pairs.count = (int*)calloc(mdd->N, sizeof(int));
//	cudaMalloc((void**)&d_pairs.count, mdd->N*sizeof(int));

	countL1Pairs_kernel<<<this->blockCount, this->blockSize>>>(d_excl.count, d_excl.list, d_pairs.count, cutoffSq, mdd->N);

	cudaMemcpy(h_pairs.count, d_pairs.count, mdd->N*sizeof(int), cudaMemcpyDeviceToHost);

	int maxPairsPerAtom = 0;
	for(i = 0; i < mdd->N; i++){
		if(maxPairsPerAtom < h_pairs.count[i]){
			maxPairsPerAtom = h_pairs.count[i];
		}
	}
	maxPairsPerAtom += PAIRLIST_EXTENSION;

	h_pairs.list = (int*)calloc(mdd->widthTot*maxPairsPerAtom, sizeof(int));
	cudaMalloc((void**)&d_pairs.list, mdd->widthTot*maxPairsPerAtom*sizeof(int));
	timesUpdated = 0;

	updateL1Pairs_kernel<<<this->blockCount, this->blockSize>>>(d_excl.count, d_excl.list, d_pairs.count, d_pairs.list,
			d_old_coord, cutoffSq, mdd->N);
}

PairListL1::~PairListL1(){
	free(h_excl.count);
	free(h_excl.list);
//	free(h_pairs.count);
	free(h_pairs.list);
	cudaFree(d_excl.count);
	cudaFree(d_excl.list);
//	cudaFree(d_pairs.count);
	cudaFree(d_pairs.list);
}


inline void PairListL1::doUpdate(MDData *mdd) {
	updateL1Pairs_kernel<<<this->blockCount, this->blockSize>>>(d_excl.count, d_excl.list, d_pairs.count, d_pairs.list,
						d_old_coord, cutoffSq, mdd->N);
}
