#pragma once

class HarmonicFixation : public IPotential {
public:
	HarmonicFixation(MDData* mdd, float3* fixedAtomsR0, float* ks);
	~HarmonicFixation();
	void compute();
	int getEnergyCount(){return 1;}
	std::string getEnergyName(int energyId){return "HarmonicFixation";}
	float getEnergies(int energyId, int timestep);
private:
	MDData* mdd;

	float3* h_fixedAtomsR0;
	float* h_ks;
	float3* d_fixedAtomsR0;
	float* d_ks;

	float* h_fmod;
	float* d_fmod;

	float averFmod;

	float* h_energy;
	float* d_energy;
};
