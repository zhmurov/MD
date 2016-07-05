/*
 * Langevin.cuh
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#pragma once

class Langevin : public IPotential {
public:
	Langevin(MDData *mdd);
	~Langevin();
	void compute(MDData *mdd);
	int get_energy_count() {return 0;}
	std::string get_energy_name(int energy_id) {return "Langevin";}
	float get_energies(int energy_id, int timestep) {return 0.0f;}
private:
	float gamma;
	float var;
};

