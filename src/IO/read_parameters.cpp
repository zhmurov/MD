/*
 * read_parameters.cpp
 *
 *  Created on: 21.08.2012
 *      Author: zhmurov
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "read_parameters.h"

#define MAXLINE 256

ReadParameters::ReadParameters(char *filename, ReadTopology *top)
{
	allocated = false;
	bond_coeffs_flag = angle_coeffs_flag = dihedral_coeffs_flag = improper_coeffs_flag = 0;
	gauss_coeffs_flag = lj_excl_coeffs = 0;

	this->top = top;

	allocate();
	read_parameters(filename);
}

ReadParameters::~ReadParameters()
{
	if (allocated) {
		delete [] bond_coeffs;
		delete [] angle_coeffs;
		delete [] dihedral_coeffs;
		delete [] improper_coeffs;

		int i;
		for (i=0;i<top->natom_types;i++) {
			delete [] ex_gauss_coeffs[i];
		}
		delete [] ex_gauss_coeffs;
	}
}

void ReadParameters::allocate()
{
	bond_coeffs = new Coeffs[top->nbond_types];
	angle_coeffs = new Coeffs[top->nangle_types];
	dihedral_coeffs = new Coeffs[top->nbond_types];
	improper_coeffs = new Coeffs[top->nangle_types];

	int n = top->natom_types;
	ex_gauss_coeffs = new Excl_Gauss_Coeffs*[n];
	for (int i=0;i<n;i++) {
		ex_gauss_coeffs[i] = new Excl_Gauss_Coeffs[n];
	}

	allocated = true;
}

void ReadParameters::read_parameters(char *filename)
{
	int n;
	char *line, *ptr, *keyword;

	line = new char[MAXLINE];

	FILE *file;
	file = fopen(filename,"r");

	while (1) {
		// Read file line by line
		if (fgets(line,MAXLINE,file) == NULL) break;

		// trim anything after '#'
		// if line is blank, continue
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) continue;

		keyword = trim(line);

		if (strcmp(keyword,"Bond Coeffs")==0) {
			nbond_types = read_coeffs(file, bond_coeffs, top->nbond_types, "bond");
			bond_coeffs_flag = 1;
		} else if (strcmp(keyword,"Angle Coeffs")==0) {
			nangle_types = read_coeffs(file, angle_coeffs, top->nangle_types, "angle");
			angle_coeffs_flag = 1;
		} else if (strcmp(keyword,"Dihedral Coeffs")==0) {
			ndihedral_types = read_coeffs(file, dihedral_coeffs, top->ndihedral_types, "dihedral");
			dihedral_coeffs_flag = 1;
		} else if (strcmp(keyword,"Improper Coeffs")==0) {
			nimproper_types = read_coeffs(file, improper_coeffs, top->nimproper_types, "improper");
			improper_coeffs_flag = 1;
		} else if (strcmp(keyword,"Gauss Coeffs")==0) {
			read_gauss_coeffs(file, ex_gauss_coeffs, top->natom_types);
			gauss_coeffs_flag = 1;
		} else if (strcmp(keyword,"LJ_Repulsive Coeffs")==0) {
			read_lj_excl_coeffs(file, ex_gauss_coeffs, top->natom_types);
			lj_excl_coeffs = 1;
		}
	}

	delete [] line;
	fclose(file);
}

int ReadParameters::read_coeffs(FILE *file, Coeffs *coeffs, int ncoeffs, const char *str)
{
	int nread, m;
	char *line, *ptr, *tmp;

	line = new char[MAXLINE];

	vector <string> values;

	// Skip 1st line
	char *eof = fgets(line,MAXLINE,file);
	if (eof == NULL) {
		print_error("Unexpected end of data file.\n");
	}

	nread = 0;
	while (1)
	{
		// Read file line by line
		if (fgets(line,MAXLINE,file) == NULL) break;

		// trim anything after '#'
		// if line is blank, continue
		if (strspn(line," \t\n\r") == strlen(line)) break;
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';

		if (nread>=ncoeffs) print_error("Incorrect number of %s coefficients in parameter file \n", str);

		m = 0;
		tmp = strtok(line," \t\n\r\f");
//		vector <string> values;
		values.clear();
		while (tmp!=NULL) {
			m++;

			values.push_back(tmp);
			tmp = strtok(NULL," \t\n\r\f");
		}
		if (m<=2) print_error("Incorrect description of a %s type\n", str);
		coeffs[nread].id = atoi(values.at(0).c_str());
		coeffs[nread].name = values.at(1);
		coeffs[nread].coeffs.assign(values.begin()+2,values.end());

		nread++;
	}

	if (nread!=ncoeffs) print_error("Incorrect number of %s coefficients in parameter file \n", str);

	delete []  line;

	return nread;
}

void ReadParameters::read_gauss_coeffs(FILE *file, Excl_Gauss_Coeffs **coeffs, int natom_types)
{
	int numG=-1, i, j, iG;
	double B_val,C_val,R_val;

	int file_state, narg;
	char ln[MAXLINE], *line, *arg[10], *ptr;
	enum File_States{FS_NONE=0, FS_GAUSS, FS_GAUSS_BCR};

	// Skip 1st line
	char *eof = fgets(ln,MAXLINE,file);
	if (eof == NULL) {
		print_error("Unexpected end of data file.\n");
	}

	file_state = FS_GAUSS;
	while ( fgets ( ln, sizeof ln, file ) != NULL ) {
		line = trim(ln);

		if (numG>0 && file_state!=FS_GAUSS_BCR) print_error("Pair_style Excl/Gauss: Error reading parameter file\n");

		// trim anything after '#'
		// if line is blank, continue
		if (strspn(line," \t\n\r") == strlen(line)) break;
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) continue;

		narg = 0;
		arg[narg] = strtok(line," \t\n");
		while ( arg[narg]!=NULL ) {
				narg++;
				if (narg>3) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Gauss coeff)\n");
				arg[narg] = strtok(NULL," \t\n");
		}

		if (narg!=3) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Gauss coeff)\n");

		if (file_state==FS_GAUSS) {
			i = atoi(arg[0])-1;
			j = atoi(arg[1])-1;
			if (i<0 || i>=natom_types || j<0 || j>=natom_types) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Gauss coeff)\n");
			numG = atoi(arg[2]);
			iG = 0;

			// Allocate B, C and R arrays
			coeffs[i][j].ng = numG;
			if (!coeffs[i][j].B) {
				coeffs[i][j].B = new double[numG];
				coeffs[i][j].C = new double[numG];
				coeffs[i][j].R = new double[numG];
			}

			if (i!=j) {
				coeffs[j][i].ng = numG;
				if (!coeffs[j][i].B) {
					coeffs[j][i].B = new double[numG];
					coeffs[j][i].C = new double[numG];
					coeffs[j][i].R = new double[numG];
				}
			}

			file_state=FS_GAUSS_BCR;
		} else if (file_state==FS_GAUSS_BCR) {
			if (numG<=0) print_error("Pair_style Excl/Gauss: Error reading coefficient file\n");

			B_val = atof(arg[0]);
			C_val = atof(arg[1]);
			R_val = atof(arg[2]);

			coeffs[i][j].B[iG] = B_val;
			coeffs[i][j].C[iG] = C_val;
			coeffs[i][j].R[iG] = R_val;

			if (i!=j) {
				coeffs[j][i].B[iG] = B_val;
				coeffs[j][i].C[iG] = C_val;
				coeffs[j][i].R[iG] = R_val;
			}

			numG--;
			iG++;
		}
		if (numG==0) file_state=FS_GAUSS;
	}
}

void ReadParameters::read_lj_excl_coeffs(FILE *file, Excl_Gauss_Coeffs **coeffs, int natom_types)
{
	int i, j;
	double A_val,l_val;

	int narg;
	char ln[MAXLINE], *line, *arg[10], *ptr;
	enum File_States{FS_NONE=0, FS_GAUSS, FS_GAUSS_BCR};

	// Skip 1st line
	char *eof = fgets(ln,MAXLINE,file);
	if (eof == NULL) {
		print_error("Unexpected end of data file.\n");
	}

	while ( fgets ( ln, sizeof ln, file ) != NULL ) {
		line = trim(ln);

		// trim anything after '#'
		// if line is blank, continue
		if (strspn(line," \t\n\r") == strlen(line)) break;
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) continue;

		narg = 0;
		arg[narg] = strtok(line," \t\n");
		while ( arg[narg]!=NULL ) {
			narg++;
			if (narg>4) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Excluded Volume coeff)\n");
			arg[narg] = strtok(NULL," \t\n");
		}
		if (narg!=4) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Excluded Volume coeff)\n");

		i = atoi(arg[0])-1;
		j = atoi(arg[1])-1;
		if (i<0 || i>=natom_types || j<0 || j>=natom_types) print_error("Pair_style Excl/Gauss: Wrong format in coefficient file (Excluded Volume coeff)\n");

		A_val = atof(arg[2]);
		l_val = atof(arg[3]);

		coeffs[i][j].ex_flag = true;
		coeffs[i][j].A = A_val;
		coeffs[i][j].l = l_val;

		if (i!=j) {
			coeffs[j][i].ex_flag = true;
			coeffs[j][i].A = A_val;
			coeffs[j][i].l = l_val;
		}
	}
}


char *ReadParameters::trim(char *str)
{
	char *newstr=new char[MAXLINE];
	int start = strspn(str," \t\n\r");
	int stop = strlen(str) - 1;
	while (str[stop] == ' ' || str[stop] == '\t'
		 || str[stop] == '\n' || str[stop] == '\r') stop--;
	str[stop+1] = '\0';
	strcpy(newstr,&str[start]);

	return newstr;
}

void ReadParameters::print_error(const char *format, ...)
{
	va_list vars;
	va_start(vars, format);
	vprintf(format, vars);
	va_end(vars);
	exit(0);
}
