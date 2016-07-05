/*
 * parameters.h
 *
 *  Created on: 16.08.2012
 *      Author: zhmurov
 */

#pragma once

#define FILENAME_LENGTH						256
#define PARAMETER_MAX_LENGTH					256

#define PARAMETER_GPU_DEVICE					"device"

#define PARAMETER_TOPOLOGY_FILENAME				"topology_file"
#define PARAMETER_PARAMETERS_FILENAME				"parameter_file"
#define PARAMETER_COORDINATES_FILENAME				"coordinates_file"

#define PARAMETER_TIMESTEP					"timestep"
#define PARAMETER_NUMSTEPS					"run"
#define PARAMETER_TEMPERATURE					"temperature"
#define PARAMETER_RSEED							"seed"

#define PARAMETER_INTEGRATOR					"integrator"
#define VALUE_INTEGRATOR_LEAP_FROG				"leap-frog"
#define VALUE_INTEGRATOR_VELOCITY_VERLET			"velocity-verlet"
#define VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER			"leap-frog-nose-hoover"

#define PARAMETER_NOSE_HOOVER_TAU				"nose-hoover-tau"
#define PARAMETER_NOSE_HOOVER_T0				"nose-hoover-T0"

#define PARAMETER_FIX_MOMENTUM					"fix_momentum"
#define DEFAULT_FIX_MOMENTUM					0
#define PARAMETER_FIX_MOMENTUM_FREQUENCE			"fix_momentum_freq"

#define PARAMETER_LANGEVIN					"langevin"
#define DEFAULT_LANGEVIN					0
#define PARAMETER_LANGEVIN_SEED					"langevin_seed"
#define PARAMETER_DAMPING					"damping"

#define PARAMETER_EXCLUDE_BOND_TYPES				"exclude_bond_types"
#define PARAMETER_PAIRLIST_CUTOFF				"pairs_cutoff"
#define PARAMETER_PAIRLIST_FREQUENCE				"pairs_freq"
#define PARAMETER_POSSIBLE_PAIRLIST_CUTOFF			"possible_pairs_cutoff"
#define PARAMETER_POSSIBLE_PAIRLIST_FREQUENCE			"possible_pairs_freq"
#define PARAMETER_NONBONDED_CUTOFF				"nb_cutoff"

#define PARAMETER_DIELECTRIC					"dielectric"
#define DEFAULT_DIELECTRIC					1.0
#define PARAMETER_PPPM_ORDER					"pppm_order"
#define DEFAULT_PPPM_ORDER					5
#define PARAMETER_PPPM_ACCURACY					"pppm_accuracy"
#define PARAMETER_COULOMB_CUTOFF				"coul_cutoff"

#define PARAMETER_PSF_OUTPUT_FILENAME				"psf_filename"
#define PARAMETER_DCD_OUTPUT_FILENAME				"dcd_filename"
#define PARAMETER_DCD_OUTPUT_FREQUENCY				"dcd_freq"
#define PARAMETER_ENERGY_OUTPUT_FREQUENCY			"energy_freq"
#define PARAMETER_ENERGY_OUTPUT_FILENAME			"energy_filename"

