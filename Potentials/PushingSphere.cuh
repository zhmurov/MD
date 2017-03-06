/*
 * PushingSphere.cu
 *
 *  Created on: 17.10.2016
 *      Author: kir_min
 */

#pragma once

#include "math.h"

class PushingSphere : public IPotential {
public:
	PushingSphere(MDData *mdd, float R0, float R, float4 centerPoint, int updatefreq, float sigma, float epsilon, const char* outdatfilename);
	~PushingSphere();
	void compute();
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "PushingSphere";}
	float get_energies(int energy_id, int timestep);
private:
	float R0;
	float R;
	float4 centerPoint;
	int updatefreq;
	float sigma;
	float epsilon;
	float* h_p_sphere;
	float* d_p_sphere;
	char filename[1024];
	float* h_energy;
	float* d_energy;
};
