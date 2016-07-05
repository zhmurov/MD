/*
 * PairListL2.cu
 *
 *  Created on: 29.08.2012
 *      Author: zhmurov
 */


#include "PairListL2.cuh"

PairListL2::PairListL2(MDData *mdd, PairlistData d_pairsL1, float cutoff, float cuton, int frequence) : PairlistUpdater(mdd)
{
	this->frequence = frequence;
	this->mdd = mdd;
	//	blockCount = (mdd->N)/DEFAULT_BLOCK_SIZE + 1;
	//	blockSize = DEFAULT_BLOCK_SIZE;
	this->cutoff = cutoff;//getFloatParameter(PARAMETER_PAIRLIST_CUTOFF);
	cutoffSq = cutoff*cutoff;
	//float nbCutoff = getFloatParameter(PARAMETER_NONBONDED_CUTOFF);
	float diff = 0.5f*(cutoff - cuton);
	drsq_up = diff*diff;
	this->frequence = frequence;//getIntegerParameter(PARAMETER_PAIRLIST_FREQUENCE);

	this->d_pairsL1 = d_pairsL1;

	countL2Pairs_kernel<<<this->blockCount, this->blockSize>>>(d_pairsL1.count, d_pairsL1.list, d_pairs.count, cutoffSq, mdd->N);

	cudaMemcpy(h_pairs.count, d_pairs.count, mdd->N*sizeof(int), cudaMemcpyDeviceToHost);

	int maxPairsPerAtom = 0;
	for(int i = 0; i < mdd->N; i++){
		if(maxPairsPerAtom < h_pairs.count[i]){
			maxPairsPerAtom = h_pairs.count[i];
		}
	}
	maxPairsPerAtom += PAIRLIST_EXTENSION;

	h_pairs.list = (int*)calloc(mdd->widthTot*maxPairsPerAtom, sizeof(int));
	cudaMalloc((void**)&d_pairs.list, mdd->widthTot*maxPairsPerAtom*sizeof(int));
	timesUpdated = 0;

	updateL2Pairlist_kernel<<<this->blockCount, this->blockSize>>>(d_pairsL1.count, d_pairsL1.list, d_pairs.count, d_pairs.list,
			d_old_coord, cutoffSq, mdd->N);
}

PairListL2::~PairListL2(){
//	free(h_pairs.count);
	free(h_pairs.list);
//	cudaFree(d_pairs.count);
	cudaFree(d_pairs.list);
}

inline void PairListL2::doUpdate(MDData *mdd) {
	updateL2Pairlist_kernel<<<this->blockCount, this->blockSize>>>(d_pairsL1.count, d_pairsL1.list, d_pairs.count, d_pairs.list,
						d_old_coord, cutoffSq, mdd->N);
}
