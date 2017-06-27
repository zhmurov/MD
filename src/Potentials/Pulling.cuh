#pragma once

class Pulling : public IPotential {
public:
	Pulling(MDData* mdd, float3* h_base_r0, int base_freq, float vel, float3* h_n, float* h_ks, int dcd_freq);
	~Pulling();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Pulling";}
	float getEnergies(int energyId, int timestep);
private:
	float base_displacement;

	int base_freq;
	float vel;
	int dcd_freq;

	float3* h_base_r0;
	float3* d_base_r0;

	float3* h_n;
	float3* d_n;

	float* d_ks;
	float* h_ks;

	float* h_fmod;
	float* d_fmod;

	float aver_fmod;

	float* h_energy;
	float* d_energy;
};
	
