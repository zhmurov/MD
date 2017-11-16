/*
 * PairListL2.cuh
 *
 *  Created on: 29.08.2012
 *      Author: zhmurov
 */

#pragma once

#include "PairlistUpdater.cuh"

class PairListL2 : public PairlistUpdater {
public:
	PairListL2(MDData *mdd, PairlistData d_pairsL1, float cutoff, float cuton, int frequence, int pairListExtension);
	~PairListL2();
	inline void doUpdate(MDData *mdd);
private:
	PairlistData d_pairsL1;
};
