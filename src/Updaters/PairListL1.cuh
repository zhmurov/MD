/*
 * PairListL1.cuh
 *
 *  Created on: 29.08.2012
 *      Author: zhmurov
 */

#pragma once

#include "PairlistUpdater.cuh"

class PairListL1 : public PairlistUpdater {
public:
	PairListL1(MDData *mdd, std::vector<int2> exclusions, float cutoff, float cuton, int frequence, int pairListExtension);
	~PairListL1();
	inline void doUpdate(MDData *mdd);
private:
	PairlistData h_excl;
	PairlistData d_excl;
};
