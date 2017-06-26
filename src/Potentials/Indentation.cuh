#pragma once

#include "../IO/xyzio.h"
#include "../IO/dcdio.h"

class Indentation : public IPotential {
public:
	Indentation(MDData *mdd, int N, int tip_radius, float3 tip_coord, float3 base_coord, int base_freq, float3 n, float vel, float ks, float eps, float sigm, float3 sf_coord, float3 sf_n, float sf_eps, float sf_sigm, int dcd_freq, char* dcd_cant_filename);
	~Indentation();
	void compute();
	int getEnergyCount(){return 0;}
	std::string getEnergyName(int energyId){return "Indentation";}
	float getEnergies(int energyId, int timestep);
private:
	int N;

	float tip_radius;
	float3 tip_coord;
	float3 base_coord;
	int base_freq;
	float3 n;
	float vel;
	float ks;
	float eps;
	float sigm;
	float3 sf_coord;
	float3 sf_n;
	float sf_eps;
	float sf_sigm;
	int dcd_freq;
	char* dcd_cant_filename;

	float const1;		// -A*x0 - B*y0 - C*z0
	float const2;		// sqrt(A^2 + B^2 + C^2)

	float tip_displacement;
	float base_displacement;

	float3 tip_current_coord;
	int current_step;

	float3* h_forcetip;
	float3* d_forcetip;

	float* h_energy;
	float* d_energy;

	DCD dcd_cant;
};
