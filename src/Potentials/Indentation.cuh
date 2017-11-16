#pragma once

#include "../IO/xyzio.h"
#include "../IO/dcdio.h"

#define FILENAME_LENGTH		256

class Indentation : public IPotential {
public:
	Indentation(MDData *mdd, int atomCount, int tipRadius, float3 tipCoord, float tipFriction, float3 baseCoord, int baseFreq, float3 baseDir, float baseVel, float ks, float eps, float sigm, float3 sfCoord, float3 sfN, float sfEps, float sfSigm, int dcdFreq, char* dcdCantFilename, char* indOutputFilename);
	~Indentation();
	void compute();
	int getEnergyCount(){return 0;}
	std::string getEnergyName(int energyId){return "Indentation";}
	float getEnergies(int energyId, int timestep);
private:
	MDData* mdd;
	int atomCount;
	float tipRadius;
	float3 tipCoord;
	float tipFriction;
	float3 baseCoord;
	int baseFreq;
	float3 baseDir;
	float baseVel;
	float ks;
	float eps;
	float sigm;
	float3 sfCoord;
	float3 sfN;
	float sfEps;
	float sfSigm;
	int dcdFreq;
	char* dcdCantFilename;

	char outputFilename[FILENAME_LENGTH];

	float const1;		// -A*x0 - B*y0 - C*z0
	float const2;		// sqrt(A^2 + B^2 + C^2)

	float tipDisplacement;
	float baseDisplacement;
	float3 tipCurrentCoord;

	float3* h_tipForce;
	float3* d_tipForce;

	float* h_energy;
	float* d_energy;

	DCD dcd_cant;
};
