/*
 * AngleClass2.cu
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */

#include "AngleClass2.cuh"

AngleClass2::AngleClass2(MDData *mdd, ReadTopology &top, ReadParameters &par){

	this->mdd = mdd;

	// Angle types

	int i, j;
	int atCount = 0;
	int* atMap = (int*)calloc(par.nangle_types, sizeof(int));
	for(i = 0; i < par.nangle_types; i++){
		Coeffs& coeffs = par.angle_coeffs[i];
		if(coeffs.name == string(ANGLE_CLASS2_STRING)){
			atMap[i] = atCount;
			atCount ++;
		} else {
			atMap[i] = -1;
		}
	}
	h_ad.pars = (float4*)calloc(atCount, sizeof(float4));
	cudaMalloc((void**)&d_ad.pars, atCount*sizeof(float4));
	atCount = 0;
	for(i = 0; i < par.nangle_types; i++){
		Coeffs& coeffs = par.angle_coeffs[i];
		if(coeffs.name == string(ANGLE_CLASS2_STRING)){
			h_ad.pars[atCount].x = atof(coeffs.coeffs.at(0).c_str())*M_PI/180.0f; // theta0
			h_ad.pars[atCount].y = 2.0f*atof(coeffs.coeffs.at(1).c_str()); // 2*k2
			h_ad.pars[atCount].z = 3.0f*atof(coeffs.coeffs.at(2).c_str()); // 3*k3
			h_ad.pars[atCount].w = 4.0f*atof(coeffs.coeffs.at(3).c_str()); // 4*k4
			atCount ++;
		}
	}

	// Angles
	angleCount = 0;
	lastAngled = 0;
	int* angleTypes = (int*)calloc(top.nangles, sizeof(int));
	for(i = 0; i < top.nangles; i++){
		angleTypes[i] = -1;
		for(j = 0; j < par.nangle_types; j++){
			if(top.angles[i].type == par.angle_coeffs[j].id && par.angle_coeffs->name == string(ANGLE_CLASS2_STRING)){
				angleTypes[i] = atMap[j];
			}
		}
		if(angleTypes[i] != -1){
			angleCount ++;
			int a1 = top.angles[i].atom1-1;
			int a2 = top.angles[i].atom2-1;
			int a3 = top.angles[i].atom3-1;
			if(a1 > lastAngled){
				lastAngled = a1;
			}
			if(a2 > lastAngled){
				lastAngled = a2;
			}
			if(a3 > lastAngled){
				lastAngled = a3;
			}
		}
	}
	h_ad.angles = (int4*)calloc(angleCount, sizeof(int4));
	cudaMalloc((void**)&d_ad.angles, angleCount*sizeof(int4));
	h_ad.refs = (int4*)calloc(angleCount, sizeof(int4));
	cudaMalloc((void**)&d_ad.refs, angleCount*sizeof(int4));
	h_ad.count = (int*)calloc(lastAngled + 1, sizeof(int));
	cudaMalloc((void**)&d_ad.count, (lastAngled + 1)*sizeof(int));
	angleCount = 0;
	for(i = 0; i < top.nangles; i++){
		if(angleTypes[i] != -1){
			int a1 = top.angles[i].atom1-1;
			int a2 = top.angles[i].atom2-1;
			int a3 = top.angles[i].atom3-1;
			h_ad.angles[angleCount].x = a1;
			h_ad.angles[angleCount].y = a2;
			h_ad.angles[angleCount].z = a3;
			h_ad.angles[angleCount].w = angleTypes[i];
			h_ad.refs[angleCount].x = h_ad.count[a1];
			h_ad.refs[angleCount].y = h_ad.count[a2];
			h_ad.refs[angleCount].z = h_ad.count[a3];
			h_ad.count[a1]++;
			h_ad.count[a2]++;
			h_ad.count[a3]++;
			angleCount ++;
		}
	}

	blockCount = (angleCount - 1)/DEFAULT_BLOCK_SIZE + 1;
	blockSize = DEFAULT_BLOCK_SIZE;

	int maxAngles = 0;
	for(i = 0; i <= lastAngled; i++){
		if(h_ad.count[i] > maxAngles){
			maxAngles = h_ad.count[i];
		}
	}

	widthTot = ((lastAngled)/DEFAULT_DATA_ALLIGN + 1)*DEFAULT_DATA_ALLIGN;

	h_ad.forces = (float4*)calloc(maxAngles*widthTot, sizeof(float4));
	cudaMalloc((void**)&d_ad.forces, maxAngles*widthTot*sizeof(float4));

	blockCountSum = lastAngled/DEFAULT_BLOCK_SIZE + 1;
	blockSizeSum = DEFAULT_BLOCK_SIZE;

	h_ad.energies = (float*)calloc(angleCount, sizeof(float));
	cudaMalloc((void**)&d_ad.energies, angleCount*sizeof(float));

	cudaMemcpy(d_ad.angles, h_ad.angles, angleCount*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ad.refs, h_ad.refs, angleCount*sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ad.count, h_ad.count, (lastAngled + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ad.energies, h_ad.energies, angleCount*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ad.pars, h_ad.pars, atCount*sizeof(float4), cudaMemcpyHostToDevice);

	cudaBindTexture(0, t_anglePar, d_ad.pars, atCount*sizeof(float4));
}

AngleClass2::~AngleClass2(){
	free(h_ad.angles);
	free(h_ad.refs);
	free(h_ad.pars);
	free(h_ad.count);
	free(h_ad.forces);
	free(h_ad.energies);
	cudaFree(d_ad.angles);
	cudaFree(d_ad.refs);
	cudaFree(d_ad.pars);
	cudaFree(d_ad.count);
	cudaFree(d_ad.forces);
	cudaFree(d_ad.energies);
}

__global__ void angleClass2Compute_kernel(int4* d_angles, int4* d_refs, float4* d_forces, int widthTot, int angleCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < angleCount){

		int4 angle = d_angles[d_i];
		float4 r1 = tex1Dfetch(t_coord, angle.x);  // Atom i
		float4 r2 = tex1Dfetch(t_coord, angle.y);  // Atom j
		float4 r3 = tex1Dfetch(t_coord, angle.z);  // Atom k
		float4 par = tex1Dfetch(t_anglePar, angle.w); // Potential parameters

		float3 dr12, dr32;

		dr12.x = r1.x - r2.x;
		dr12.y = r1.y - r2.y;
		dr12.z = r1.z - r2.z;

		dr32.x = r3.x - r2.x;
		dr32.y = r3.y - r2.y;
		dr32.z = r3.z - r2.z;

		//Apply PBC

		float3 pb = c_mdd.bc.len;

		dr12.x -= rint(dr12.x/pb.x)*pb.x;
		dr12.y -= rint(dr12.y/pb.y)*pb.y;
		dr12.z -= rint(dr12.z/pb.z)*pb.z;

		dr32.x -= rint(dr32.x/pb.x)*pb.x;
		dr32.y -= rint(dr32.y/pb.y)*pb.y;
		dr32.z -= rint(dr32.z/pb.z)*pb.z;

		float r12inv = 1.0f/sqrtf(dr12.x*dr12.x + dr12.y*dr12.y + dr12.z*dr12.z);
		float r32inv = 1.0f/sqrtf(dr32.x*dr32.x + dr32.y*dr32.y + dr32.z*dr32.z);
		float costheta = (dr12.x*dr32.x + dr12.y*dr32.y + dr12.z*dr32.z)*r12inv*r32inv;

		if(costheta > 1.0f){
			costheta = 1.0f;
		} else
		if(costheta < -1.0f){
			costheta = -1.0f;
		}

		float sintheta = sqrtf(1.0f - costheta*costheta);
		float theta = acos(costheta);
		float arg = theta - par.x;  //theta - theta0

		if(sintheta < SMALL){
			sintheta = SMALL;
		}

		float diff = -arg*(par.y + arg*(par.z + arg*par.w));	// -dU/dtheta
		diff /= sintheta;										// -(dU/dtheta)/sin

		/*if(sintheta < 1.e-6){
			if(diff < 0){
				diff *= 2.0f*par.y;
			} else {
				diff *= -2.0f*par.y;
			}
		} else {
			diff *= (-2.0f*par.y) / sintheta;
		}*/

		float c1 = diff*r12inv; 								// [-(dU/dtheta)/sin]*(1/rji)
		float c2 = diff*r32inv;									// [-(dU/dtheta)/sin]*(1/rjk)

		float4 f1, f2, f3;
		f1.x = c1*(dr12.x*(r12inv*costheta) - dr32.x*r32inv);
		f1.y = c1*(dr12.y*(r12inv*costheta) - dr32.y*r32inv);
		f1.z = c1*(dr12.z*(r12inv*costheta) - dr32.z*r32inv);

		f3.x = c2*(dr32.x*(r32inv*costheta) - dr12.x*r12inv);
		f3.y = c2*(dr32.y*(r32inv*costheta) - dr12.y*r12inv);
		f3.z = c2*(dr32.z*(r32inv*costheta) - dr12.z*r12inv);

		f2.x = -f1.x-f3.x;
		f2.y = -f1.y-f3.y;
		f2.z = -f1.z-f3.z;

		int4 ref = d_refs[d_i];

		d_forces[widthTot*ref.x + angle.x] = f1;
		d_forces[widthTot*ref.y + angle.y] = f2;
		d_forces[widthTot*ref.z + angle.z] = f3;
	}
}

__global__ void angleClass2Sum_kernel(int* d_count, float4* d_forces, int widthTot, int lastAngled){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i <= lastAngled){
		float4 f = c_mdd.d_force[d_i];
		int j;
		for(j = 0; j < d_count[d_i]; j++){
			float4 df = d_forces[j*widthTot + d_i];
			f.x += df.x;
			f.y += df.y;
			f.z += df.z;
		}
		c_mdd.d_force[d_i] = f;
	}
}

void AngleClass2::compute(MDData *mdd){
	angleClass2Compute_kernel<<<this->blockCount, this->blockSize>>>(d_ad.angles, d_ad.refs, d_ad.forces, widthTot, angleCount);
	//cudaThreadSynchronize();
	angleClass2Sum_kernel<<<blockCountSum, blockSizeSum>>>(d_ad.count, d_ad.forces, widthTot, lastAngled);
	/*int i;
	cudaMemcpy(mdd->h_force, mdd->d_force, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	FILE* file = fopen("angle_forces.dat", "w");
	for(i = 0; i < mdd->N; i++){
		fprintf(file, "%f %f %f\n", mdd->h_force[i].x, mdd->h_force[i].y, mdd->h_force[i].z);
	}
	fclose(file);
	exit(0);*/
}


__global__ void angleClass2Energy_kernel(int4* d_angles, float* d_energies, int angleCount){
	int d_i = blockIdx.x*blockDim.x + threadIdx.x;
	if(d_i < angleCount){

		int4 angle = d_angles[d_i];
		float4 r1 = tex1Dfetch(t_coord, angle.x);  // Atom i
		float4 r2 = tex1Dfetch(t_coord, angle.y);  // Atom j
		float4 r3 = tex1Dfetch(t_coord, angle.z);  // Atom k
		float4 par = tex1Dfetch(t_anglePar, angle.w); // Potential parameters

		float3 dr12, dr32;

		dr12.x = r1.x - r2.x;
		dr12.y = r1.y - r2.y;
		dr12.z = r1.z - r2.z;

		dr32.x = r3.x - r2.x;
		dr32.y = r3.y - r2.y;
		dr32.z = r3.z - r2.z;

		//Apply PBC

		float3 pb = c_mdd.bc.len;

		dr12.x -= rint(dr12.x/pb.x)*pb.x;
		dr12.y -= rint(dr12.y/pb.y)*pb.y;
		dr12.z -= rint(dr12.z/pb.z)*pb.z;

		dr32.x -= rint(dr32.x/pb.x)*pb.x;
		dr32.y -= rint(dr32.y/pb.y)*pb.y;
		dr32.z -= rint(dr32.z/pb.z)*pb.z;

		float r12inv = 1.0f/sqrtf(dr12.x*dr12.x + dr12.y*dr12.y + dr12.z*dr12.z);
		float r32inv = 1.0f/sqrtf(dr32.x*dr32.x + dr32.y*dr32.y + dr32.z*dr32.z);
		float costheta = (dr12.x*dr32.x + dr12.y*dr32.y + dr12.z*dr32.z)*r12inv*r32inv;

		if(costheta > 1.0f){
			costheta = 1.0f;
		} else
		if(costheta < -1.0f){
			costheta = -1.0f;
		}

		float theta = acos(costheta);
		float arg = theta - par.x;  //theta - theta0

		float pot = arg*arg*(0.5f*par.y + arg*(par.z/3.0f + 0.25*arg*par.w));	// U

		d_energies[d_i] = pot;
	}
}


float AngleClass2::get_energies(int energy_id, int timestep){
	angleClass2Energy_kernel<<<this->blockCount, this->blockSize>>>(d_ad.angles, d_ad.energies, angleCount);
	//cudaThreadSynchronize();
	cudaMemcpy(h_ad.energies, d_ad.energies, angleCount*sizeof(float), cudaMemcpyDeviceToHost);
	int i;
	float energy = 0.0f;
	for(i = 0; i < angleCount; i++){
		energy += h_ad.energies[i];
	}
	return energy;
}
