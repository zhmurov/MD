#pragma once

#define FILENAME_LENGTH		256

class Pulling : public IPotential {
public:
	Pulling(MDData* mdd, float3* baseR0, int baseFreq, float vel, float3* n, float* ks, int dcdFreq, char* pullOutputFilename);
	~Pulling();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "Pulling";}
	float getEnergies(int energyId, int timestep);
private:
	MDData* mdd;
	int baseFreq;
	float vel;
	int dcdFreq;

	char outputFilename[FILENAME_LENGTH];
	
	float baseDisplacement;

	float3* h_baseR0;
	float3* h_n;
	float* h_ks;
	float3* d_baseR0;
	float3* d_n;
	float* d_ks;

	float* h_fmod;
	float* d_fmod;

	float averFmod;

	float* h_energy;
	float* d_energy;
};
