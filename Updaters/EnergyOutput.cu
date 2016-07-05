/*
 * EnergyOutput.cu
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#include "EnergyOutput.cuh"

EnergyOutput::EnergyOutput(MDData *mdd, std::vector<IPotential*>* potentials){
	this->mdd = mdd;
	this->frequence = getIntegerParameter(PARAMETER_ENERGY_OUTPUT_FREQUENCY);
	this->potentials = potentials;
	getMaskedParameter(filename, PARAMETER_ENERGY_OUTPUT_FILENAME);
	FILE* file = fopen(filename, "w");
	fclose(file);
}

EnergyOutput::~EnergyOutput(){

}

void EnergyOutput::update(MDData *mdd){
	int p, i;
	FILE* file = fopen(filename, "a");

	//if(mdd->step%(this->frequence*10) == 0){
		printf("%*s%*s",
				ENERGY_OUTPUT_WIDTH, ENERGY_OUTPUT_STEP,
				ENERGY_OUTPUT_WIDTH, ENERGY_OUTPUT_TEMPERATURE);
		for(p = 0; p != (*potentials).size(); p++){
			for(i = 0; i < (*potentials)[p]->get_energy_count(); i++){
				printf("%*s", ENERGY_OUTPUT_WIDTH, (*potentials)[p]->get_energy_name(i).c_str());
			}
		}
		printf("%*s\n", ENERGY_OUTPUT_WIDTH, ENERGY_OUTPUT_TOTAL);
	//}


	double temp = 0.0f;
	cudaMemcpy(mdd->h_vel, mdd->d_vel, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	for(i = 0; i < mdd->N; i++){
		temp += mdd->h_vel[i].w*mdd->h_mass[i];
		mdd->h_vel[i].w = 0.0f;
	}
	cudaMemcpy(mdd->d_vel, mdd->h_vel, mdd->N*sizeof(float4), cudaMemcpyHostToDevice);
	temp /= ((float)mdd->N)*((float)this->frequence)*3.0f*BOLTZMANN_CONSTANT;
//	temp /= ((float)mdd->N)*3.0f*BOLTZMANN_CONSTANT;

	printf("%*d%*f",
			ENERGY_OUTPUT_WIDTH, mdd->step,
			ENERGY_OUTPUT_WIDTH, temp);
	fprintf(file, "%d\t%f\t", mdd->step, temp);
	float totalEnergy = 0;
	for(p = 0; p != (*potentials).size(); p++){
		for(i = 0; i < (*potentials)[p]->get_energy_count(); i++){
			float energy = (*potentials)[p]->get_energies(i, mdd->step);
			totalEnergy += energy;
			printf("%*f", ENERGY_OUTPUT_WIDTH, energy);
			fprintf(file, "%f\t", energy);
		}
	}
	printf("%*f\n", ENERGY_OUTPUT_WIDTH, totalEnergy);
	fprintf(file, "%f\n", totalEnergy);
	fclose(file);
}



