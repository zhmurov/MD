#include "Indentation.cuh"

Indentation::Indentation(MDData *mdd, int base_freq, float3 base_coord, float tip_radius, float3 tip_coord, float3 n, float vel, float ks, float eps, float sigm, int dcd_freq, char* dcd_cant_filename, float3 plane, float const1){

	this->mdd = mdd;
	this->base_freq = base_freq;
	this->base_coord = base_coord;
	this->tip_radius = tip_radius;
	this->tip_coord = tip_coord;
	this->n = n;
	this->vel = vel;
	this->ks = ks;
	this->eps = eps;
	this->sigm = sigm;
	this->dcd_cant_filename = dcd_cant_filename;
	this->const1 = const1;

	this->const2 = const2;
	const2 = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
	this->tip_displacement = tip_displacement;
	tip_displacement = 0.0;
	this->base_displacement = base_displacement;
	base_displacement = 0.0;
	this->tip_current_coord = tip_current_coord;
	tip_current_coord = tip_coord;
	this->current_step = current_step;
	current_step = 0;

	FILE *txt = fopen("data.out", "w");
	fclose(txt);

//DCD
	this->dcd_freq = dcd_freq;

	int frameCount = mdd->numsteps/dcd_freq + 1;
	createDCD(&dcd_cant, 2, frameCount, 1, current_step, dcd_freq, 1, mdd->bc.len.x, mdd->bc.len.y, mdd->bc.len.z);
	dcdOpenWrite(&dcd_cant, dcd_cant_filename);
	dcdWriteHeader(dcd_cant);

	this->blockCount = (mdd->N-1)/DEFAULT_BLOCK_SIZE + 1;
	this->blockSize = DEFAULT_BLOCK_SIZE;

//FORCE
	h_forcetip = (float3*)calloc(mdd->N, sizeof(float3));
	cudaMalloc((void**)&d_forcetip, mdd->N*sizeof(float3));
	cudaMemcpy(d_forcetip, h_forcetip, mdd->N*sizeof(float3), cudaMemcpyHostToDevice);

//ENERGY
	h_energy = (float*)calloc(mdd->N, sizeof(float));
	cudaMalloc((void**)&d_energy, mdd->N*sizeof(float));
	cudaMemcpy(d_energy, h_energy, mdd->N*sizeof(float), cudaMemcpyHostToDevice);
}

Indentation::~Indentation(){
	free(h_forcetip);
	free(h_energy);
	cudaFree(d_forcetip);
	cudaFree(d_energy);
	//TODO destroyDCD(&dcd);
}

__global__ void Indentation_kernel(float tip_radius, float3 tip_current_coord, float eps, float sigm, float3* d_forcetip, float3 n, float const1, float const2){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		float temp;
		float rij_mod, df;
		float3 rij, rj;
		float4 ri, f;

		ri = c_mdd.d_coord[i];
		rj = tip_current_coord;

		f = c_mdd.d_force[i];

		rij.x = rj.x - ri.x;
		rij.y = rj.y - ri.y;
		rij.z = rj.z - ri.z;

		rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

	//REPULSIVE POTENTIAL
	//BIDS
		temp = 6.0f*pow(sigm, 6.0f)/pow((rij_mod - tip_radius), 7.0f);

		df = -eps*temp/rij_mod;

		f.x += df*rij.x;
		f.y += df*rij.y;
		f.z += df*rij.z;

	//TIP
		d_forcetip[i].x = -df*rij.x;
		d_forcetip[i].y = -df*rij.y;
		d_forcetip[i].z = -df*rij.z;

	//PLANE
		rij_mod = abs(n.x*ri.x + n.y*ri.y + n.z*ri.z + const1)/const2;

		temp = 6.0f*pow(sigm, 6.0f)/pow(rij_mod, 7.0f);

		df = -eps*temp/rij_mod;

		f.x += df*rij_mod*n.x;
		f.y += df*rij_mod*n.y;
		f.z += df*rij_mod*n.z;


		c_mdd.d_force[i] = f;
	}
}


void Indentation::compute(){

	current_step++;

	tip_current_coord.x = tip_coord.x + tip_displacement*n.x;
	tip_current_coord.y = tip_coord.y + tip_displacement*n.y;
	tip_current_coord.z = tip_coord.z + tip_displacement*n.z;


	Indentation_kernel<<<this->blockCount, this->blockSize>>>(tip_radius, tip_current_coord, eps, sigm, d_forcetip, n, const1, const2);


	//TODO current_step -> mdd->step

	if (current_step % base_freq == 0){
		base_displacement = vel*mdd->dt*current_step;
	}

	cudaMemcpy(h_forcetip, d_forcetip, mdd->N*sizeof(float3), cudaMemcpyDeviceToHost);

	float3 resforce = make_float3(0.0f, 0.0f, 0.0f);
	for(int i = 0; i < mdd->N; i++){
		resforce.x += h_forcetip[i].x;
		resforce.y += h_forcetip[i].y;
		resforce.z += h_forcetip[i].z;
	}

	if (current_step % dcd_freq == 0){
		printf("resforce.z = %f\n", resforce.z);
	}

	float mult = -ks*(tip_displacement - base_displacement);

	resforce.x += n.x*mult;
	resforce.y += n.y*mult;
	resforce.z += n.z*mult;

	// FRICTION COEFFICIENT
	// ksi = 6*pi*nu*r = 5.655E+4
	// 1/ksi = 1.77E-5

	tip_displacement += 0.0001*mdd->dt*(resforce.x*n.x + resforce.y*n.y + resforce.z*n.z);


	//TODO current_step -> mdd->step
	if (current_step % dcd_freq == 0){

		FILE *txt = fopen("data.out", "a");
		fprintf(txt, "%3.6f\t", (base_coord.z + base_displacement*n.z)*10.0);
		fprintf(txt, "%3.6f\n", (tip_coord.z + tip_displacement*n.z)*10.0);
		fclose(txt);

		printf("%3.6f\t%3.6f\n", (base_coord.z + base_displacement*n.z)*10.0, (tip_coord.z + tip_displacement*n.z)*10.0);

		//tip
		dcd_cant.frame.X[0] = (tip_coord.x + tip_displacement*n.x)*10.0;			// [nm] -> [angstr]
		dcd_cant.frame.Y[0] = (tip_coord.y + tip_displacement*n.y)*10.0;			// [nm] -> [angstr]
		dcd_cant.frame.Z[0] = (tip_coord.z + tip_displacement*n.z)*10.0;			// [nm] -> [angstr]
		//base
		dcd_cant.frame.X[1] = (base_coord.x + base_displacement*n.x)*10.0;	// [nm] -> [angstr]
		dcd_cant.frame.Y[1] = (base_coord.y + base_displacement*n.y)*10.0;	// [nm] -> [angstr]
		dcd_cant.frame.Z[1] = (base_coord.z + base_displacement*n.z)*10.0;	// [nm] -> [angstr]

		dcdWriteFrame(dcd_cant);
	}
}

__global__ void Indentation_Energy_kernel(float tip_radius, float3 tip_current_coord, float eps, float sigm, float* d_energy){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < c_mdd.N){

		float rij_mod, energy;
		float3 rij, rj;
		float4 ri;

		ri = c_mdd.d_coord[i];
		rj = tip_current_coord;

		rij.x = rj.x - ri.x;
		rij.y = rj.y - ri.y;
		rij.z = rj.z - ri.z;

		rij_mod = sqrt(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

		//REPULSIVE POTENTIAL
		energy = eps*pow(sigm, 6.0f)/pow((rij_mod - tip_radius), 6.0f);

		d_energy[i] = energy;
	}
}



float Indentation::get_energies(int energy_id, int Nstep){

	Indentation_Energy_kernel<<<this->blockCount, this->blockSize>>>(tip_radius, tip_current_coord, eps, sigm, d_energy);

	cudaMemcpy(h_energy, d_energy, mdd->N*sizeof(float), cudaMemcpyDeviceToHost);
	float energy_sum = 0.0;

	for (int i = 0; i < mdd->N; i++){
		energy_sum += h_energy[i];
	}
	return energy_sum;
}
