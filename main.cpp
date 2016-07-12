/*
 * main.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: Aram Davtyan, Artem Zhmurov
 */

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "IO/read_topology.h"
#include "IO/read_parameters.h"
#include "IO/configreader.h"
#include "IO/psfio.h"
#include "IO/xyzio.h"
#include "parameters.h"
//#include "Util/ReductionAlgorithms.h"

using namespace std;

extern void compute();

int main(int argc, char *argv[])
{
	compute();
	destroyConfigReader();
	return 0;
}

