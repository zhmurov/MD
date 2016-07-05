/*
 * PPPM.cu
 *
 *  Created on: 30.08.2012
 *      Author: zhmurov
 */

#include "PPPM.cuh"
#include "../IO/pdbio.h"

#define EPS_HOC 1e-7
#define MAX_ORDER 7
#define MAX_MESH_SIZE 16384

#define LARGE 10000.0

PPPM::PPPM(MDData *mdd, float dielectric, float cutoff)
{
	// Initialize parameters

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;
	this->mdd = mdd;

	lastStepEnergyComputed = -1;

	pb = mdd->bc.len;

	printf("\nInitializing PPPM...\n");

	order = getIntegerParameter(PARAMETER_PPPM_ORDER, DEFAULT_PPPM_ORDER);
	accuracy = getFloatParameter(PARAMETER_PPPM_ACCURACY);
	this->cutoff = cutoff;

	this->dielectric = dielectric;
	q_factor = QQR2E/dielectric;

	if (order>MAX_ORDER) DIE("ERROR: PPPM order is too big.");

	factors = NULL;
	acons = NULL;

	// Calculating qsum and qsumsq
	qsum = 0.0f;
	qsumsq = 0.0f;
	for(int i = 0; i < mdd->N; i++){
		qsum += mdd->h_charge[i];
		qsumsq += mdd->h_charge[i]*mdd->h_charge[i];
	}
	printf("Total charge of the system: %f\n", qsum);

	set_grid_parameters();

	printf("Alpha parameter: %f\n", alpha);
	printf("Mesh size: %d %d %d\n\n", mesh_size.x, mesh_size.y, mesh_size.z);


	mesh_dim = mesh_size.x*mesh_size.y*mesh_size.z;

	this->blockCountMesh = (mesh_dim-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSizeMesh = DEFAULT_BLOCK_SIZE;

	// Allocate

	h_rho_coeff = (float*)calloc(order*order, sizeof(float));
	cudaMalloc((void**)&d_rho_coeff, order*order*sizeof(float));

	h_mesh = (cufftComplex*)calloc(mesh_dim, sizeof(cufftComplex));
	cudaMalloc((void**)&d_mesh, mesh_dim*sizeof(cufftComplex));

	h_Ex = (cufftComplex*)calloc(mesh_dim, sizeof(cufftComplex));
	h_Ey = (cufftComplex*)calloc(mesh_dim, sizeof(cufftComplex));
	h_Ez = (cufftComplex*)calloc(mesh_dim, sizeof(cufftComplex));
	cudaMalloc((void**)&d_Ex, mesh_dim*sizeof(cufftComplex));
	cudaMalloc((void**)&d_Ey, mesh_dim*sizeof(cufftComplex));
	cudaMalloc((void**)&d_Ez, mesh_dim*sizeof(cufftComplex));

	h_El = (float3*)calloc(mesh_dim, sizeof(float3));
	cudaMalloc((void**)&d_El, mesh_dim*sizeof(float3));

	h_k_vec = (float3*)calloc(mesh_dim, sizeof(float3));
	cudaMalloc((void**)&d_k_vec, mesh_dim*sizeof(float3));

	h_G = (float*)calloc(mesh_dim, sizeof(float));
	cudaMalloc((void**)&d_G, mesh_dim*sizeof(float));

	h_energies = (float*)calloc(mesh_dim, sizeof(float));
	cudaMalloc((void**)&d_energies, mesh_dim*sizeof(float));

	h_gf_b = (double*)calloc(order, sizeof(double));
	cudaMalloc((void**)&d_gf_b, order*sizeof(double));

	// Initialize arrays

	compute_polynomial_coefficients();

	cudaMemcpy(d_rho_coeff, h_rho_coeff, order*order*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mesh, h_mesh, mesh_dim*sizeof(cufftComplex), cudaMemcpyHostToDevice);

	compute_gf_denom_coefficients();

	cudaMemcpy(d_gf_b, h_gf_b, order*sizeof(double), cudaMemcpyHostToDevice);

	int3 nb;
	nb.x = floor( (alpha*pb.x/(M_PI*mesh_size.x)) * pow(-log(EPS_HOC),0.25) );
	nb.y = floor( (alpha*pb.y/(M_PI*mesh_size.y)) * pow(-log(EPS_HOC),0.25) );
	nb.z = floor( (alpha*pb.z/(M_PI*mesh_size.z)) * pow(-log(EPS_HOC),0.25) );

	calculate_kvec_green_hat_kernel<<<blockCountMesh, blockSizeMesh>>>(d_k_vec, d_G, d_gf_b, alpha, order, mesh_size, nb);

	cufftPlan3d(&plan, mesh_size.x, mesh_size.y, mesh_size.z, CUFFT_C2C);

}

PPPM::~PPPM()
{
	free(h_rho_coeff);
	free(h_mesh);
	free(h_Ex);
	free(h_Ey);
	free(h_Ez);
	free(h_El);
	free(h_k_vec);
	free(h_G);
	free(h_energies);
	free(h_gf_b);

	cudaFree(d_rho_coeff);
	cudaFree(d_mesh);
	cudaFree(d_Ex);
	cudaFree(d_Ey);
	cudaFree(d_Ez);
	cudaFree(d_El);
	cudaFree(d_k_vec);
	cudaFree(d_G);
	cudaFree(d_energies);
	cudaFree(d_gf_b);

	cufftDestroy(plan);

	if (factors!=NULL) delete [] factors;
	if (acons!=NULL) {
		free(acons[0]);
		free(acons);
	}
}

void PPPM::compute(MDData *mdd)
{
	cudaMemset(d_mesh, 0, sizeof(cufftComplex)*mesh_dim);

	// Spread charges on the mesh
	assign_charges_to_grid_kernel<<<blockCount, blockSize>>>(d_mesh, d_rho_coeff, order, mesh_size);

	cudaThreadSynchronize();

	cufftExecC2C(plan, d_mesh, d_mesh, CUFFT_FORWARD);

	cudaThreadSynchronize();

	// Calculate field components Ex Ey Ez by multiplying d_mesh by green function
	multiply_by_green_function_kernel<<<blockCountMesh, blockSizeMesh>>>(d_Ex, d_Ey, d_Ez, d_mesh, d_k_vec, d_G, mesh_dim);

	cudaThreadSynchronize();

	cufftExecC2C(plan, d_Ex, d_Ex, CUFFT_INVERSE);
	cufftExecC2C(plan, d_Ey, d_Ey, CUFFT_INVERSE);
	cufftExecC2C(plan, d_Ez, d_Ez, CUFFT_INVERSE);

	cudaThreadSynchronize();

	// El = (Ex, Ey, Ez)
	reassign_electric_field_kernel<<<blockCountMesh, blockSizeMesh>>>(d_Ex, d_Ey, d_Ez, d_El, mesh_dim);

	cudaThreadSynchronize();

	// Force calculation
	compute_PPPM_forces_kernel<<<blockCount, blockSize>>>(d_El, d_rho_coeff, order, mesh_size, q_factor);

	/*int i;
	cudaThreadSynchronize();
	cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	FILE* file = fopen("pppm_forces.dat", "w");
	for(i = 0; i < mdd->N; i++){
		fprintf(file, "%f %f %f\n", mdd->h_force[i].x, mdd->h_force[i].y, mdd->h_force[i].z);
	}
	fclose(file);
	exit(0);*/
}

float PPPM::get_energies(int energy_id, int timestep)
{
	if(timestep != lastStepEnergyComputed){
		compute_PPPM_energy_kernel<<<this->blockCountMesh, this->blockSizeMesh>>>(d_energies, d_mesh, d_G, mesh_dim);

		cudaMemcpy(h_energies, d_energies, mesh_dim*sizeof(float), cudaMemcpyDeviceToHost);

		energyValue = 0.0f;
		for(int i = 0; i < mesh_dim; i++){
			energyValue += h_energies[i];
		}
		lastStepEnergyComputed = timestep;

		float s2 = 1.0f/(float)mesh_dim;
		s2 = s2*s2;
		energyValue *= s2;

		float volume = pb.x*pb.y*pb.z;
		energyValue *= 0.5f*volume;
		energyValue -= alpha*qsumsq/sqrt(M_PI);;
		energyValue -= M_PI_2*qsum*qsum/(alpha*alpha*volume);
		energyValue *= QQR2E/dielectric;

	}
	return energyValue;
}

void PPPM::compute_polynomial_coefficients()
{
	int i,j,k,l;
	float **a, s;
	a = new float*[order];
	for (i=0; i<order; i++) {
		a[i] = new float[2*order+1];
		a[i] += order;
	}

	for (i=0; i<order; i++) {
		for (j= -order; j<= order; j++) {
			a[i][j] = 0.0;
		}
	}

	a[0][0] = 1.0;
	for (j=1; j<order; j++) {
		for (k= -j; k<= j; k += 2) {
			s = 0.0;
			for (l = 0; l < j; l++) {
				a[l+1][k] = (a[l][k+1]-a[l][k-1]) / (l+1);
				s += powf(0.5,(float) l+1) *
					  (a[l][k-1] + powf(-1.0,(float) l) * a[l][k+1]) / (l+1);
			}
			a[0][k] = s;
		}
	}

	i = 0;
	for (k = -(order-1); k < order; k += 2) {
		for (l = 0; l < order; l++) {
			h_rho_coeff[l*order+i] = a[l][k];
		}
		i++;
	}

	for (i=0; i<order; i++) {
		delete [] &a[i][-order];
	}

	delete [] a;
}

void PPPM::compute_gf_denom_coefficients()
{
	int k,l,m;

	for (l = 1; l < order; l++) h_gf_b[l] = 0.0f;
	h_gf_b[0] = 1.0f;

	for (m = 1; m < order; m++) {
		for (l = m; l > 0; l--)
			h_gf_b[l] = 4.0f * (h_gf_b[l]*(l-m)*(l-m-0.5f)-h_gf_b[l-1]*(l-m-1)*(l-m-1));
		h_gf_b[0] = 4.0f * (h_gf_b[0]*(l-m)*(l-m-0.5f));
	}

	int ifact = 1;
	for (k = 1; k < 2*order; k++) ifact *= k;
	float gaminv = 1.0f/(float)ifact;
	for (l = 0; l < order; l++) h_gf_b[l] *= gaminv;
}

void PPPM::set_grid_parameters()
{
	if (accuracy <= 0.0) DIE("ERROR: PPPM accuracy must be positive.");

	accuracy *= QQR2E;

	nfactors = 3;
	factors = new int[nfactors];
	factors[0] = 2;
	factors[1] = 3;
	factors[2] = 5;

	acons = new double*[8];
	for (int i=0;i<8;i++) {
		acons[i] = new double[7];
	}

	acons[1][0] = 2.0 / 3.0;
	acons[2][0] = 1.0 / 50.0;
	acons[2][1] = 5.0 / 294.0;
	acons[3][0] = 1.0 / 588.0;
	acons[3][1] = 7.0 / 1440.0;
	acons[3][2] = 21.0 / 3872.0;
	acons[4][0] = 1.0 / 4320.0;
	acons[4][1] = 3.0 / 1936.0;
	acons[4][2] = 7601.0 / 2271360.0;
	acons[4][3] = 143.0 / 28800.0;
	acons[5][0] = 1.0 / 23232.0;
	acons[5][1] = 7601.0 / 13628160.0;
	acons[5][2] = 143.0 / 69120.0;
	acons[5][3] = 517231.0 / 106536960.0;
	acons[5][4] = 106640677.0 / 11737571328.0;
	acons[6][0] = 691.0 / 68140800.0;
	acons[6][1] = 13.0 / 57600.0;
	acons[6][2] = 47021.0 / 35512320.0;
	acons[6][3] = 9694607.0 / 2095994880.0;
	acons[6][4] = 733191589.0 / 59609088000.0;
	acons[6][5] = 326190917.0 / 11700633600.0;
	acons[7][0] = 1.0 / 345600.0;
	acons[7][1] = 3617.0 / 35512320.0;
	acons[7][2] = 745739.0 / 838397952.0;
	acons[7][3] = 56399353.0 / 12773376000.0;
	acons[7][4] = 25091609.0 / 1560084480.0;
	acons[7][5] = 1755948832039.0 / 36229939200000.0;
	acons[7][6] = 4887769399.0 / 37838389248.0;

	q2 = qsumsq*QQR2E/dielectric/dielectric;

	// initial estimate of alpha based on accuracy, number of atoms and coul cutoff
	alpha = accuracy*sqrt(mdd->N*cutoff*pb.x*pb.y*pb.z) / (2.0*q2);
	if (alpha >= 1.0) alpha = (1.35 - 0.15*log(accuracy))/cutoff;
	else alpha = sqrt(-log(alpha)) / cutoff;

	double3 h;
	double err;

	h.x = h.y = h.z = 1.0/alpha;
	mesh_size.x = (int)(pb.x/h.x) + 1;
	mesh_size.y = (int)(pb.y/h.y) + 1;
	mesh_size.z = (int)(pb.z/h.z) + 1;

	err = estimate_error(h.x,pb.x);
	while (err > accuracy) {
		err = estimate_error(h.x,pb.x);
		mesh_size.x++;
		h.x = pb.x/mesh_size.x;
		printf("mesh_size: %d %d %d\n", mesh_size.x, mesh_size.y, mesh_size.z);
	}

	err = estimate_error(h.y,pb.y);
	while (err > accuracy) {
		err = estimate_error(h.y,pb.y);
		mesh_size.y++;
		h.y = pb.y/mesh_size.y;
	}

	err = estimate_error(h.z,pb.z);
	while (err > accuracy) {
		err = estimate_error(h.z,pb.z);
		mesh_size.z++;
		h.z = pb.z/mesh_size.z;
		printf("mesh_size: %d %d %d\n", mesh_size.x, mesh_size.y, mesh_size.z);
	}

	while (!factorable(mesh_size.x)) mesh_size.x++;
	while (!factorable(mesh_size.y)) mesh_size.y++;
	while (!factorable(mesh_size.z)) mesh_size.z++;

	if (mesh_size.x >= MAX_MESH_SIZE || mesh_size.y >= MAX_MESH_SIZE || mesh_size.z >= MAX_MESH_SIZE)
		DIE("ERROR: PPPM grid is too large.");

	adjust_alpha();
}

double PPPM::estimate_error(double hx, double pbx)
{
	double sum = 0.0;
	for (int m = 0; m < order; m++)
		sum += acons[order][m] * pow(hx*alpha,2.0*m);
	double value = q2 * pow(hx*alpha,(double)order) *
	sqrt(alpha*pbx*sqrt(2.0*M_PI)*sum/mdd->N) / (pbx*pbx);

	return value;
}

int PPPM::factorable(int n)
{
  int i;

  while (n > 1) {
    for (i = 0; i < nfactors; i++) {
      if (n % factors[i] == 0) {
        n /= factors[i];
        break;
      }
    }
    if (i == nfactors) return 0;
  }

  return 1;
}

/* ----------------------------------------------------------------------
   adjust the alpha parameter to near its optimal value
   using a Newton-Raphson solver
------------------------------------------------------------------------- */
void PPPM::adjust_alpha()
{
  double dx;

  for (int i = 0; i < LARGE; i++) {
    dx = newton_raphson_f() / derivf();
    alpha -= dx;
    if (fabs(newton_raphson_f()) < 0.00001) return;
  }

  DIE("ERROR: Could not compute alpha in PPPM.");
}

/* ----------------------------------------------------------------------
 Calculate f(x) using Newton-Raphson solver
 ------------------------------------------------------------------------- */

double PPPM::newton_raphson_f()
{
  double df_rspace = 2.0*q2*exp(-alpha*alpha*cutoff*cutoff) /
       sqrt(mdd->N*cutoff*pb.x*pb.y*pb.z);

  double df_kspace = compute_df_kspace();

  return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
 Calculate numerical derivative f'(x) using forward difference
 [f(x + h) - f(x)] / h
 ------------------------------------------------------------------------- */

double PPPM::derivf()
{
  double h = 0.000001;  //Derivative step-size
  double df,f1,f2,alpha_old;

  f1 = newton_raphson_f();
  alpha_old = alpha;
  alpha += h;
  f2 = newton_raphson_f();
  alpha = alpha_old;
  df = (f2 - f1)/h;

  return df;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
------------------------------------------------------------------------- */

double PPPM::compute_df_kspace()
{
	double df_kspace = 0.0;
	double lprx = estimate_error(pb.x/mesh_size.x,pb.x);
	double lpry = estimate_error(pb.y/mesh_size.y,pb.y);
	double lprz = estimate_error(pb.z/mesh_size.z,pb.z);
	df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
	return df_kspace;
}

/* ----------------------------------------------------------------------
   Calculate the final estimate of the accuracy
------------------------------------------------------------------------- */

/*double PPPM::final_accuracy()
{
  double df_kspace = compute_df_kspace();
  double q2_over_sqrt = q2 / sqrt(mdd->N*cutoff*pb.x*pb.y*pb.z);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-alpha*alpha*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace + df_table*df_table);

  return estimated_accuracy;
}*/


__device__ inline void AddToMeshPoint(int X, int Y, int Z, cufftComplex* array, float value, int size_y, int size_z)
{
	atomicAdd(&array[Z + size_z * (Y + size_y * X)].x, value);
}

__global__ void assign_charges_to_grid_kernel(cufftComplex *d_mesh, float *d_rho_coeff, int order, int3 mesh_size)
{
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_i < c_mdd.N) {
		float qi = c_mdd.d_charge[d_i];

		if (qi!=0.0f) {
			float4 ri = c_mdd.d_coord[d_i];//tex1Dfetch(t_coord, d_i);

			float3 pb = c_mdd.bc.len;
			float mesh_dx = pb.x / (float)mesh_size.x;
			float mesh_dy = pb.y / (float)mesh_size.y;
			float mesh_dz = pb.z / (float)mesh_size.z;

			ri.x += pb.x * 0.5f;
			ri.y += pb.y * 0.5f;
			ri.z += pb.z * 0.5f;

			ri.x /= mesh_dx;
			ri.y /= mesh_dy;
			ri.z /= mesh_dz;

			float shift, shiftone, x0, y0, z0, dx, dy, dz;
			int nlower, nupper, mx, my, mz, nxi, nyi, nzi;

			nlower = -(order-1)/2;
			nupper = order/2;

			if (order % 2) {
				shift = 0.5f;
				shiftone = 0.0f;
			} else {
				shift = 0.0f;
				shiftone = 0.5f;
			}

			nxi = __float2int_rd(ri.x + shift);
			nyi = __float2int_rd(ri.y + shift);
			nzi = __float2int_rd(ri.z + shift);

			dx = shiftone+(float)nxi-ri.x;
			dy = shiftone+(float)nyi-ri.y;
			dz = shiftone+(float)nzi-ri.z;

			int n,m,l,k;
			float result;

			float sum=0.0f;

			x0 = qi / (mesh_dx*mesh_dy*mesh_dz);
			for (n = nlower; n <= nupper; n++) {
				mx = n+nxi;
				if (mx >= mesh_size.x) mx -= mesh_size.x;
				if (mx < 0) mx += mesh_size.x;
				result = 0.0f;
				for (k = order-1; k >= 0; k--) {
					result = d_rho_coeff[n-nlower + k*order] + result * dx;
				}
				y0 = x0*result;
				for (m = nlower; m <= nupper; m++) {
					my = m+nyi;
					if(my >= mesh_size.y) my -= mesh_size.y;
					if(my < 0)  my += mesh_size.y;
					result = 0.0f;
					for (k = order-1; k >= 0; k--) {
						result = d_rho_coeff[m-nlower + k*order] + result * dy;
					}
					z0 = y0*result;
					for (l = nlower; l <= nupper; l++) {
						mz = l+nzi;
						if(mz >= mesh_size.z) mz -= mesh_size.z;
						if(mz < 0)  mz += mesh_size.z;
						result = 0.0f;
						for (k = order-1; k >= 0; k--) {
							result = d_rho_coeff[l-nlower + k*order] + result * dz;
						}
						sum += z0*result;
						AddToMeshPoint(mx, my, mz, d_mesh, z0*result, mesh_size.y, mesh_size.z);
					}
				}
			}
		}
	}
}

__global__ void compute_PPPM_forces_kernel(float3 *d_El, float *d_rho_coeff, int order, int3 mesh_size, float q_factor)
{
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_i < c_mdd.N) {
		float qi = c_mdd.d_charge[d_i];

		if (qi!=0.0f) {
			float4 ri = tex1Dfetch(t_coord, d_i);
			float4 force = c_mdd.d_force[d_i];

			qi *= q_factor;

			float3 pb = c_mdd.bc.len;
			float mesh_dx = pb.x / (float)mesh_size.x;
			float mesh_dy = pb.y / (float)mesh_size.y;
			float mesh_dz = pb.z / (float)mesh_size.z;

			ri.x += pb.x * 0.5f;
			ri.y += pb.y * 0.5f;
			ri.z += pb.z * 0.5f;

			ri.x /= mesh_dx;
			ri.y /= mesh_dy;
			ri.z /= mesh_dz;

			float shift, shiftone, x0, y0, z0, dx, dy, dz;
			int nlower, nupper, mx, my, mz, nxi, nyi, nzi;

			nlower = -(order-1)/2;
			nupper = order/2;

			if (order % 2) {
				shift = 0.5f;
				shiftone = 0.0f;
			} else {
				shift = 0.0f;
				shiftone = 0.5f;
			}

			nxi = __float2int_rd(ri.x + shift);
			nyi = __float2int_rd(ri.y + shift);
			nzi = __float2int_rd(ri.z + shift);

			dx = shiftone+(float)nxi-ri.x;
			dy = shiftone+(float)nyi-ri.y;
			dz = shiftone+(float)nzi-ri.z;

			int n,m,l,k;
			float result;
			float3 El;
			int mult_fact = order;

			for (n = nlower; n <= nupper; n++) {
				mx = n+nxi;
				if (mx >= mesh_size.x) mx -= mesh_size.x;
				if (mx < 0) mx += mesh_size.x;
				result = 0.0f;
				for (k = order-1; k >= 0; k--) {
					result = d_rho_coeff[n-nlower + k*mult_fact] + result * dx;
				}
				x0 = result;
				for (m = nlower; m <= nupper; m++) {
					my = m+nyi;
					if(my >= mesh_size.y) my -= mesh_size.y;
					if(my < 0)  my += mesh_size.y;
					result = 0.0f;
					for (k = order-1; k >= 0; k--) {
						result = d_rho_coeff[m-nlower + k*mult_fact] + result * dy;
					}
					y0 = x0*result;
					for (l = nlower; l <= nupper; l++) {
						mz = l+nzi;
						if(mz >= mesh_size.z) mz -= mesh_size.z;
						if(mz < 0)  mz += mesh_size.z;
						result = 0.0f;
						for (k = order-1; k >= 0; k--) {
							result = d_rho_coeff[l-nlower + k*mult_fact] + result * dz;
						}
						z0 = y0*result;
						El = d_El[mz + mesh_size.z * (my + mesh_size.y * mx)];
						force.x += qi*z0*El.x;
						force.y += qi*z0*El.y;
						force.z += qi*z0*El.z;
					}
				}
			}
			c_mdd.d_force[d_i] = force;
		}
	}
}

__global__ void multiply_by_green_function_kernel(cufftComplex* d_Ex, cufftComplex* d_Ey, cufftComplex* d_Ez,
													cufftComplex *d_mesh, float3* d_k_vec, float *d_G, int mesh_dim) {
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (d_i < mesh_dim) {
		float3 k_vec = d_k_vec[d_i];
		cufftComplex Ex, Ey, Ez;
		float factor = d_G[d_i]/((float)mesh_dim);
		cufftComplex mesh_point = d_mesh[d_i];

		mesh_point.x *= factor;
		mesh_point.y *= factor;

		Ex.x = k_vec.x * mesh_point.y;
		Ex.y = -k_vec.x * mesh_point.x;
		d_Ex[d_i] = Ex;

		Ey.x = k_vec.y * mesh_point.y;
		Ey.y = -k_vec.y * mesh_point.x;
		d_Ey[d_i] = Ey;

		Ez.x = k_vec.z * mesh_point.y;
		Ez.y = -k_vec.z * mesh_point.x;
		d_Ez[d_i] = Ez;
	}
}

__global__ void reassign_electric_field_kernel(cufftComplex *d_Ex, cufftComplex *d_Ey, cufftComplex *d_Ez,
												float3 *d_El, int mesh_dim) {
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (d_i < mesh_dim) {
		float3 El;
		El.x = d_Ex[d_i].x;
		El.y = d_Ey[d_i].x;
		El.z = d_Ez[d_i].x;

		d_El[d_i] = El;
	}
}

__device__ inline double gf_denom(double3 kh, double *gf_b, int order)
{
	double sx,sy,sz;
	double sinx,siny,sinz;

	sinx = sin(0.5*kh.x);
	siny = sin(0.5*kh.y);
	sinz = sin(0.5*kh.z);

	sinx = sinx*sinx;
	siny = siny*siny;
	sinz = sinz*sinz;

	sz = sy = sx = 0.0;
	for (int l = order-1; l >= 0; l--) {
	  sx = gf_b[l] + sx*sinx;
	  sy = gf_b[l] + sy*siny;
	  sz = gf_b[l] + sz*sinz;
	}
	double s = sx*sy*sz;
	return s*s;
}

__global__ void calculate_kvec_green_hat_kernel(float3* d_k_vec, float *d_G, double *d_gf_b, double alpha, int order, int3 mesh_size, int3 nb)
{
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (d_i < mesh_size.x*mesh_size.y*mesh_size.z) {
		double3 j, unitk;
		double k_vec_sq;
		double alpha_sq = alpha*alpha;
		int mesh_size_yz = mesh_size.y*mesh_size.z;

		int xn = d_i/mesh_size_yz;
		int yn = (d_i - xn*mesh_size_yz)/mesh_size.z;
		int zn = (d_i - xn*mesh_size_yz - yn*mesh_size.z);

		// Calculate k vectors

		j.x  = xn - mesh_size.x*(2*xn/mesh_size.x);
		j.y  = yn - mesh_size.y*(2*yn/mesh_size.y);
		j.z  = zn - mesh_size.z*(2*zn/mesh_size.z);

		float3 pb = c_mdd.bc.len;
		unitk.x = 2.0f*M_PI/pb.x;
		unitk.y = 2.0f*M_PI/pb.y;
		unitk.z = 2.0f*M_PI/pb.z;

		d_k_vec[d_i].x = j.x*unitk.x;
		d_k_vec[d_i].y = j.y*unitk.y;
		d_k_vec[d_i].z = j.z*unitk.z;

		// Calculate green hat function

		k_vec_sq = j.x*j.x*unitk.x*unitk.x + j.y*j.y*unitk.y*unitk.y + j.z*j.z*unitk.z*unitk.z;

		double W;
		double factor, sum1, denom, dot1, dot2;
		double3 q, s, w, arg, h, kh;
		int ix, iy, iz;

		h.x = pb.x/(double)mesh_size.x;
		h.y = pb.y/(double)mesh_size.y;
		h.z = pb.z/(double)mesh_size.z;

		kh.x = j.x*unitk.x*h.x;
		kh.y = j.y*unitk.y*h.y;
		kh.z = j.z*unitk.z*h.z;

		if (k_vec_sq != 0.0f) {
			factor = 4.0f*M_PI/k_vec_sq;
			denom = gf_denom(kh, d_gf_b, order);
			sum1 = 0.0f;
			for (ix = -nb.x; ix<=nb.x; ix++) {
				q.x = unitk.x*(j.x+(double)(mesh_size.x*ix));
				s.x = exp(-0.25f*q.x*q.x/alpha_sq);
				w.x = 1.0f;
				arg.x = 0.5f*q.x*h.x;
				if (arg.x!=0.0f) w.x = pow(sin(arg.x)/arg.x,order);
				for (iy = -nb.y; iy<=nb.y; iy++) {
					q.y = unitk.y*(j.y+(double)(mesh_size.y*iy));
					s.y = exp(-0.25f*q.y*q.y/alpha_sq);
					w.y = 1.0f;
					arg.y = 0.5f*q.y*h.y;
					if (arg.y!=0.0f) w.y = pow(sin(arg.y)/arg.y,order);
					for (iz = -nb.z; iz<=nb.z; iz++) {
						q.z = unitk.z*(j.z+(double)(mesh_size.z*iz));
						s.z = exp(-0.25f*q.z*q.z/alpha_sq);
						w.z = 1.0f;
						arg.z = 0.5f*q.z*h.z;
						if (arg.z!=0.0f) w.z = pow(sin(arg.z)/arg.z,order);

						dot1 = unitk.x*j.x*q.x + unitk.y*j.y*q.y + unitk.z*j.z*q.z;
						dot2 = q.x*q.x+q.y*q.y+q.z*q.z;
						W = w.x*w.y*w.z;
						sum1 += (dot1/dot2) * s.x * s.y * s.z * W*W;
					}
				}
			}
			d_G[d_i] = factor*sum1/denom;
		} else d_G[d_i] = 0.0f;
	}
}

__global__ void compute_PPPM_energy_kernel(float *d_energies, cufftComplex *d_mesh, float *d_G, int mesh_dim)
{
	int d_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (d_i < mesh_dim) {
		cufftComplex mesh = d_mesh[d_i];
		float G = d_G[d_i];
		d_energies[d_i] = G*(mesh.x*mesh.x + mesh.y*mesh.y);
	}
}

