/*
 * Langevin.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 *  Changes: 12.07.2016
 *	Author: kir_min
 */

#pragma once

class Langevin : public IPotential {
public:
	Langevin(MDData *mdd, float damping, int seed, float temperature);
	~Langevin();
	void compute();
	int get_energy_count() {return 0;}
	std::string get_energy_name(int energy_id) {return "Langevin";}
	float get_energies(int energy_id, int timestep) {return 0.0f;}
private:
	float gamma;
	float var;
};

