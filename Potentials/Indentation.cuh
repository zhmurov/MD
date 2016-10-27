#pragma once

#include "../IO/xyzio.h"
#include "../IO/dcdio.h"

class Indentation : public IPotential {
public:
	Indentation(MDData *mdd, int base_freq, float3 base_coord, float tip_radius, float3 tip_coord, float3 n, float vel, float ks, float eps, float sigm, int dcd_freq, char* dcd_tip_filename, float3 plane, float const1);
	~Indentation();
	void compute();
	int get_energy_count() {return 0;}
	std::string get_energy_name(int energy_id) {return "Indentation";}
	float get_energies(int energy_id, int timestep);
private:
	int base_freq;
	float3 base_coord;
	float tip_radius;
	float3 tip_coord;
	float3 n;
	float vel;
	float ks;
	float eps;
	float sigm;
	float3 plane;
	float const1;		// -(A*x0 + B*y0 + C*z0)
	float const2;		// sqrt(A^2 + B^2 + C^2)

	float base_displacement;
	float tip_displacement;

	float3 tip_current_coord;

	float3* h_forcetip;
	float3* d_forcetip;

	float* h_energy;
	float* d_energy;

	int current_step;
	int dcd_freq;

	char* dcd_cant_filename;

	DCD dcd_cant;
};
