/*
 * PairlistUpdater.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#pragma once

#define PAIRLIST_EXTENSION	2048

typedef struct {
	int* count;
	int* list;
} PairlistData;

class PairlistUpdater : public IUpdater {
public:
	PairlistUpdater(MDData *mdd);
	~PairlistUpdater();
	void update();
	void printPairlist();
	virtual inline void doUpdate(MDData *mdd) {};
	PairlistData h_pairs;
	PairlistData d_pairs;
protected:
	Reduction* reduction;
	float cutoff;
	float cutoffSq;
	float drsq_up;
	float *d_drsq;
	float4 *d_old_coord;
	int timesUpdated;
	char *pl_name;
	int maxPairsPerAtom;

	float rmax_displacement(float4* d_r1, float4* d_r2, int N);
};
