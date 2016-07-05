/*
 * ComputationalArrays.h
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#include "IO/read_topology.h"
#include "IO/read_parameters.h"
#include <vector>
#include <algorithm>
#include <vector_types.h>

#pragma once

class ComputationalArrays
{
public:
	ComputationalArrays(ReadTopology *top, ReadParameters *par);
	~ComputationalArrays();

	ReadTopology *top;
	ReadParameters *par;

	void GetBondList(string name, std::vector<int3> *bonds, std::vector<Coeffs> *parameters);
	void GetExclusionList(std::vector<int2> *list, std::vector<int> *excl_bond_types);
	bool QPairExcluded(int atom1, int atom2, std::vector<int> *excl_btypes);
};
