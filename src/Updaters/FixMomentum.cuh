/*
 * FixMomentum.cuh
 *
 *  Created on: 02.04.2016
 *      Author: zhmurov
 */

#pragma once

#include <Common/interfaces.h>

class FixMomentum : public IUpdater {
public:
	FixMomentum(MDData *mdd, int frequence);
	~FixMomentum();
	inline void update();
private:
	Reduction* reduction;
	ReductionFloat4* reduction4;
	float4* h_muwcoord;
	float4* d_muwcoord;
	float4* h_mvel;
	float4* d_mvel;
	float4* h_angmom;
	float4* d_angmom;
	float4* h_inertiax;
	float4* h_inertiay;
	float4* h_inertiaz;
	float4* d_inertiax;
	float4* d_inertiay;
	float4* d_inertiaz;
};
