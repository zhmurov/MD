/*
 * PPPM.cuh
 *
 *  Created on: 30.08.2012
 *      Author: zhmurov
 */

#pragma once

#include <math.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

typedef struct {
} PPPMData;

class PPPM : public IPotential {
public:
	PPPM(MDData *mdd, float dielectric, float cutoff);
	~PPPM();
	void compute();
	int get_energy_count() {return 1;}
	std::string get_energy_name(int energy_id) {return "PPPM";}
	float get_energies(int energy_id, int timestep);
	double get_alpha() {return alpha;};
private:
	int order;  // B-Spline order
	double alpha; // Spliting parameter
	double accuracy;
	double cutoff;
	float dielectric;
	float q_factor; // QQR2E/dielectric
	float qsum; // Total charge
	float qsumsq; // Square sum of all charges
	float q2; // qsumsq*QQR2E/dielectric/dielectric
	float *h_rho_coeff;  // Polynomial coefficients
	float *d_rho_coeff;
	int3 mesh_size; // Mesh dimensions
	int mesh_dim;  // Mesh dimensions product
	cufftComplex *h_mesh;
	cufftComplex *d_mesh;
	cufftComplex *d_Ex, *d_Ey, *d_Ez;
	cufftComplex *h_Ex, *h_Ey, *h_Ez;
	float3 *h_El;
	float3 *d_El;
	float3* d_k_vec;
	float3* h_k_vec;
	float *d_G; // Green function hat
	float *h_G;
	float *d_energies; // PPPM energies (per mesh point)
	float *h_energies;
	double energyValue;
	cufftHandle plan;
	double *h_gf_b;
	double *d_gf_b;

	float3 pb;

	double **acons;
	int nfactors;
	int *factors;

	int blockCountMesh;
	int blockSizeMesh;

	void compute_polynomial_coefficients();
	void compute_gf_denom_coefficients();
	void set_grid_parameters();
	double estimate_error(double hx, double pbx);
	int factorable(int n);
	void adjust_alpha();
	double newton_raphson_f();
	double derivf();
	double compute_df_kspace();
	double final_accuracy();
	inline float gf_denom(float3 kh);
};

__device__ inline void AddToMeshPoint(int X, int Y, int Z, cufftComplex* array, float value, int size_y, int size_z);
__global__ void assign_charges_to_grid_kernel(cufftComplex *d_mesh, float *d_rho_coeff, int order, int3 mesh_size);
__global__ void compute_PPPM_forces_kernel(float3 *d_El, float *d_rho_coeff, int order, int3 mesh_size, float q_factor);
__global__ void multiply_by_green_function_kernel(cufftComplex* d_Ex, cufftComplex* d_Ey, cufftComplex* d_Ez,
													cufftComplex *d_mesh, float3* d_k_vec, float *d_G, int mesh_dim);
__global__ void reassign_electric_field_kernel(cufftComplex *d_Ex, cufftComplex *d_Ey, cufftComplex *d_Ez,
												float3 *d_El, int mesh_dim);
__global__ void calculate_kvec_green_hat_kernel(float3* d_k_vec, float *d_G, double *d_gf_b, double alpha, int order, int3 mesh_size, int3 nb);
__global__ void compute_PPPM_energy_kernel(float *d_energies, cufftComplex *d_mesh, float *d_G, int mesh_dim);
