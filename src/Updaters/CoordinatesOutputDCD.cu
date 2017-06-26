/*
 * CoordinatesOutputDCD.cu
 *
 *  Created on: 16.08.2012
 *      Author: zhmurov
 */

#include "CoordinatesOutputDCD.cuh"

CoordinatesOutputDCD::CoordinatesOutputDCD(MDData *mdd){
	this->mdd = mdd;
	this->frequence = getIntegerParameter(PARAMETER_DCD_OUTPUT_FREQUENCY);
	int frameCount = mdd->numsteps/this->frequence + 1;
	float timeStep = mdd->step;
	createDCD(&dcd, mdd->N, frameCount, 1, timeStep, this->frequence, 1, mdd->bc.len.x, mdd->bc.len.y, mdd->bc.len.z);
	char filename[FILENAME_LENGTH];
	getMaskedParameter(filename, PARAMETER_DCD_OUTPUT_FILENAME);
	dcdOpenWrite(&dcd, filename);
	dcdWriteHeader(dcd);
}

CoordinatesOutputDCD::~CoordinatesOutputDCD(){
	destroyDCD(&dcd);
}

void CoordinatesOutputDCD::update(){
	cudaMemcpy(mdd->h_coord, mdd->d_coord, mdd->N*sizeof(float4), cudaMemcpyDeviceToHost);
	int i;
	for(i = 0; i < mdd->N; i++){
		dcd.frame.X[i] = mdd->h_coord[i].x*10.0;	// [nm] -> [angstr]
		dcd.frame.Y[i] = mdd->h_coord[i].y*10.0;	// [nm] -> [angstr]
		dcd.frame.Z[i] = mdd->h_coord[i].z*10.0;	// [nm] -> [angstr]
	}
	dcdWriteFrame(dcd);
}
