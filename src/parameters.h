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
#define PARAMETER_PDB_REFERENCE_FILENAME			"pdbref_file"

#define PARAMETER_POTENTIAL_BOND_FENE				"fene"
#define DEFAULT_POTENTIAL_BOND_FENE				0
#define PARAMETER_FUNCTIONTYPE_BOND_FENE			"fene_func"
#define	DEFAULT_FUNCTIONTYPE_BOND_FENE				40
#define PARAMETER_BOND_FENE_KS					"fene_ks"
#define DEFAULT_BOND_FENE_KS					-1
#define PARAMETER_BOND_FENE_R					"fene_R"

#define PARAMETER_POTENTIAL_LENNARDJONES			"lennardjones"
#define DEFAULT_POTENTIAL_LENNARDJONES				0
#define PARAMETER_FUNCTIONTYPE_LENNARDJONES			"lennardjones_func"
#define DEFAULT_FUNCTIONTYPE_LENNARDJONES			40

#define PARAMETER_POTENTIAL_REPULSIVE				"repulsive"
#define DEFAULT_POTENTIAL_REPULSIVE				0
#define PARAMETER_FUNCTIONTYPE_REPULSIVE			"repulsive_func"
#define DEFAULT_FUNCTIONTYPE_REPULSIVE				40
#define PARAMETER_REPULSIVE_EPSILON				"repulsive_eps"
#define PARAMETER_REPULSIVE_SIGMA				"repulsive_sigm"

#define PARAMETER_POTENTIAL_BONDSCLASS2ATOM			"bondclass2atom"
#define PARAMETER_POTENTIAL_ANGLECLASS2				"angleclass2"
#define PARAMETER_POTENTIAL_GAUSSEXCLUDED			"gaussexcluded"
#define PARAMETER_POTENTIAL_PPPM				"pppm"
#define PARAMETER_POTENTIAL_COULOMB				"coulomb"

#define DEFAULT_POTENTIAL_BONDSCLASS2ATOM			0
#define DEFAULT_POTENTIAL_ANGLECLASS2				0
#define DEFAULT_POTENTIAL_GAUSSEXCLUDED				0
#define DEFAULT_POTENTIAL_PPPM					0
#define DEFAULT_POTENTIAL_COULOMB				0

#define PARAMETER_FUNCTIONTYPE_BONDSCLASS2ATOM			"func_bc2a"
#define PARAMETER_FUNCTIONTYPE_ANGLECLASS2			"func_ac2"

#define DEFAULT_FUNCTIONTYPE_BONDSCLASS2ATOM			10
#define DEFAULT_FUNCTIONTYPE_ANGLECLASS2			10

#define PARAMETER_TIMESTEP					"timestep"
#define PARAMETER_NUMSTEPS					"run"
#define PARAMETER_TEMPERATURE					"temperature"
#define PARAMETER_RSEED						"seed"

#define PARAMETER_INTEGRATOR					"integrator"
#define VALUE_INTEGRATOR_LEAP_FROG				"leap-frog"
#define VALUE_INTEGRATOR_VELOCITY_VERLET			"velocity-verlet"
#define VALUE_INTEGRATOR_LEAP_FROG_NOSE_HOOVER			"leap-frog-nose-hoover"
#define VALUE_INTEGRATOR_LEAP_FROG_NEW				"leap-frog-new"
#define VALUE_INTEGRATOR_LEAP_FROG_LANGEVIN			"leap-frog-langevin"
#define PARAMETER_LEAP_FROG_LANGEVIN_DAMPING			"leap-frog-langevin-damping"
#define VALUE_INTEGRATOR_LEAP_FROG_OVERDUMPED			"leap-frog-overdumped"
#define PARAMETER_LEAP_FROG_OVERDUMPED_FRICTION			"overdumped_friction"
#define DEFAULT_LEAP_FROG_OVERDUMPED_FRICTION			4300
#define VALUE_INTEGRATOR_STEEPEST_DESCENT			"steepest-descent"
#define PARAMETER_STEEPEST_DESCENT_FRICTION			"steepestdes_friction"
#define PARAMETER_STEEPEST_DESCENT_MAXFORCE			"steepestdes_maxforce"

#define PARAMETER_FIX						"fixation"
#define DEFAULT_FIX						0

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
#define PARAMETER_PAIRLIST_EXTESION				"pairlist_extension"
#define DEFAULT_PAIRLIST_EXTESION				2048

#define PARAMETER_PUSHING_SPHERE				"push_sphere"
#define DEFAULT_PUSHING_SPHERE					0
#define PARAMETER_PUSHING_SPHERE_RADIUS0			"ps_radius0"
#define PARAMETER_PUSHING_SPHERE_RADIUS				"ps_radius"
#define PARAMETER_PUSHING_SPHERE_SPEED				"ps_speed"
#define PARAMETER_PUSHING_SPHERE_CENTER_POINT			"ps_center_of_sphere"
#define PARAMETER_PUSHING_SPHERE_UPDATE_FREQ			"ps_update_freq"
#define PARAMETER_PUSHING_SPHERE_SIGMA				"ps_sigma"
#define PARAMETER_PUSHING_SPHERE_EPSILON			"ps_epsilon"
#define PARAMETER_PUSHING_SPHERE_HARMONIC			"ps_harmonic"
#define DEFAULT_PUSHING_SPHERE_HARMONIC				0
#define PARAMETER_PUSHING_SPHERE_OUTPUT_FILENAME		"ps_presure_filename"
#define PARAMETER_PUSHING_SPHERE_MASK				"ps_mask"
#define DEFAULT_PUSHING_SPHERE_MASK				0
#define PARAMETER_PUSHING_SPHERE_MASK_PDB_FILENAME		"ps_mask_pdb_filename"

#define PARAMETER_MOTOR						"motor"
#define DEFAULT_MOTOR						0
#define PARAMETER_MOTOR_RADIUS0					"mp_radius0"
#define PARAMETER_MOTOR_FORCE					"mp_force"
#define PARAMETER_MOTOR_RADIUS_HOLE				"mp_radius_hole"
#define PARAMETER_MOTOR_HEIGHT					"mp_height"
#define PARAMETER_MOTOR_CENTER_POINT				"mp_center_of_sphere"
#define PARAMETER_MOTOR_UPDATE_FREQ				"mp_update_freq"
#define PARAMETER_MOTOR_SIGMA					"mp_sigma"
#define PARAMETER_MOTOR_EPSILON					"mp_epsilon"
#define PARAMETER_MOTOR_OUTPUT_FILENAME				"mp_presure_filename"
#define PARAMETER_MOTOR_MASK					"mp_mask"
#define DEFAULT_MOTOR_MASK							0
#define PARAMETER_MOTOR_MASK_PDB_FILENAME			"mp_mask_pdb_filename"

#define PARAMETER_INDENTATION					"indentation"
#define DEFAULT_INDENTATION					0
#define PARAMETER_INDENTATION_ATOMTYPE				"ind_atomtype"
#define PARAMETER_INDENTATION_TIP_RADIUS			"ind_tip_radius"
#define PARAMETER_INDENTATION_TIP_FRICTION			"ind_tip_friction"
#define PARAMETER_INDENTATION_TIP_COORD				"ind_tip_coord"
#define PARAMETER_INDENTATION_BASE_COORD			"ind_base_coord"
#define PARAMETER_INDENTATION_BASE_DISPLACEMENT_FREQUENCY	"ind_base_freq"
#define PARAMETER_INDENTATION_BASE_DIRECTION			"ind_base_direction"
#define PARAMETER_INDENTATION_BASE_VELOCITY			"ind_base_vel"
#define PARAMETER_INDENTATION_KSPRING				"ind_ks"
#define PARAMETER_INDENTATION_EPSILON				"ind_eps"
#define PARAMETER_INDENTATION_SIGMA				"ind_sigm"

#define PARAMETER_PULLING					"pulling"
#define DEFAULT_PULLING						0
#define PARAMETER_PULLING_BASE_DISPLACEMENT_FREQUENCY		"pulling_base_freq"
#define PARAMETER_PULLING_VELOCITY				"pulling_vel"

#define PARAMETER_SURFACE_COORD					"sf_coord"
#define PARAMETER_SURFACE_N					"sf_n"
#define PARAMETER_SURFACE_EPSILON				"sf_eps"
#define PARAMETER_SURFACE_SIGMA					"sf_sigm"

#define PARAMETER_BOND_HARMONIC					"harmonic"
#define DEFAULT_BOND_HARMONIC					0
#define PARAMETER_FUNCTIONTYPE_BOND_HARMONIC			"harmonic_func"
#define	DEFAULT_FUNCTIONTYPE_BOND_HARMONIC			40
#define PARAMETER_BOND_HARMONIC_KS				"harmonic_ks"
#define DEFAULT_BOND_HARMONIC_KS				-1

#define PARAMETER_DIELECTRIC					"dielectric"
#define DEFAULT_DIELECTRIC					1.0
#define PARAMETER_PPPM_ORDER					"pppm_order"
#define DEFAULT_PPPM_ORDER					5
#define PARAMETER_PPPM_ACCURACY					"pppm_accuracy"
#define PARAMETER_COULOMB_CUTOFF				"coul_cutoff"

#define PARAMETER_PSF_OUTPUT_FILENAME				"psf_filename"
#define PARAMETER_DCD_OUTPUT_FILENAME				"dcd_filename"
#define PARAMETER_PULLING_OUTPUT_FILENAME			"pull_output_filename"
#define PARAMETER_INDENTATION_OUTPUT_FILENAME			"ind_output_filename"
#define PARAMETER_PDB_CANTILEVER_OUTPUT_FILENAME		"pdb_cant_filename"
#define PARAMETER_DCD_CANTILEVER_OUTPUT_FILENAME		"dcd_cant_filename"
#define PARAMETER_DCD_OUTPUT_FREQUENCY				"dcd_freq"
#define PARAMETER_ENERGY_OUTPUT_FREQUENCY			"energy_freq"
#define PARAMETER_ENERGY_OUTPUT_FILENAME			"energy_filename"

#define DEFAULT_OUTPUT_XYZ					0
#define PARAMETER_OUTPUT_XYZ					"output_xyz"
#define PARAMETER_OUTPUT_XYZ_FILENAME				"output_xyz_file"
