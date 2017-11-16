/*
 * read_topology.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: aram
 */

#include "read_topology.h"

#define MAXLINE 256

ReadTopology::ReadTopology(char *filename)
{
	allocated = false;
	atoms_flag = bonds_flag = angles_flag = dihedrals_flag = impropers_flag = masses_flag = 0;

	read_topology(filename);

	if (!atoms_flag) print_error("No Atoms section in data file\n");
	if (!masses_flag) print_error("No Masses section in data file\n");
	if (nbonds>0 && !bonds_flag) print_error("No Bonds section in data file\n");
//	if (nbonds>0 && !bond_coeffs_flag) print_error("No Bonds section in data file\n");
	if (nangles>0 && !angles_flag) print_error("No Angles section in data file\n");
	if (ndihedrals>0 && !dihedrals_flag) print_error("No Dihedrals section in data file\n");
	if (nimpropers>0 && !impropers_flag) print_error("No Impropers section in data file\n");
}

ReadTopology::~ReadTopology()
{
	if (allocated) {
		delete [] atoms;
		delete [] bonds;
		delete [] angles;
		delete [] dihedrals;
		delete [] impropers;
		delete [] masses;
	}
}

void ReadTopology::allocate()
{
	atoms = new Atom[natoms];
	bonds = new Bond[nbonds];
	angles = new Angle[nangles];
	dihedrals = new Dihedral[ndihedrals];
	impropers = new Improper[nimpropers];
	masses = new Mass[natom_types];

	allocated = true;
}

void ReadTopology::read_topology(char *filename)
{
	int n;
	char *line, *ptr, *keyword;

	line = new char[MAXLINE];

	FILE *file;
	file = fopen(filename,"r");

	// Skip 1st line
	char *eof = fgets(line,MAXLINE,file);
	if (eof == NULL) {
		print_error("Unexpected end of data file.\n");
	}

	while (1) {
		// Read file line by line
		if (fgets(line,MAXLINE,file) == NULL) n = 0;
		else n = strlen(line) + 1;

		// if n = 0 then end-of-file so return with blank line
		if (n == 0) {
			line[0] = '\0';
			break;
		}

		// trim anything after '#'
		// if line is blank, continue
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) continue;

		// search line for header keyword and set corresponding variable
		if (strstr(line,"atoms")) sscanf(line,"%d",&natoms);
		else if (strstr(line,"bonds")) sscanf(line,"%d",&nbonds);
		else if (strstr(line,"angles")) sscanf(line,"%d",&nangles);
		else if (strstr(line,"dihedrals")) sscanf(line,"%d", &ndihedrals);
		else if (strstr(line,"impropers")) sscanf(line,"%d", &nimpropers);

		else if (strstr(line,"atom types")) sscanf(line,"%d",&natom_types);
		else if (strstr(line,"bond types")) sscanf(line,"%d",&nbond_types);
		else if (strstr(line,"angle types")) sscanf(line,"%d",&nangle_types);
		else if (strstr(line,"dihedral types")) sscanf(line,"%d",&ndihedral_types);
		else if (strstr(line,"improper types")) sscanf(line,"%d",&nimproper_types);

		else if (strstr(line,"xlo xhi")) sscanf(line,"%lg %lg",&box.xlo,&box.xhi);
		else if (strstr(line,"ylo yhi")) sscanf(line,"%lg %lg",&box.ylo,&box.yhi);
		else if (strstr(line,"zlo zhi")) sscanf(line,"%lg %lg",&box.zlo,&box.zhi);
		else break;
	}

	// Sanity checks
	if (natoms<=0) print_error("Number of atoms must be positive\n");
	if (natom_types<=0) print_error("Number of atom types must be positive\n");
	if (nbonds>0 && nbond_types<=0) print_error("Number of bond types must be positive if bonds are defined\n");
	if (nangles>0 && nangle_types<=0) print_error("Number of angle types must be positive if angles are defined\n");
	if (ndihedrals>0 && ndihedral_types<=0) print_error("Number of dihedral types must be positive if dihedrals are defined\n");
	if (nimpropers>0 && nimproper_types<=0) print_error("Number of improper types must be positive if impropers are defined\n");
	if (box.xlo>=box.xhi || box.ylo>=box.yhi || box.zlo>=box.zhi) print_error("Wrong definition of the simulation box\n");

	allocate();

	while (1) {
		keyword = trim(line);

		if (strcmp(keyword,"Atoms")==0) {
			read_atoms(file);
			atoms_flag = 1;
		} else if (strcmp(keyword,"Bonds")==0) {
			read_bonds(file);
			bonds_flag = 1;
		} else if (strcmp(keyword,"Angles")==0) {
			read_angles(file);
			angles_flag = 1;
		} else if (strcmp(keyword,"Dihedrals")==0) {
			read_dihedrals(file);
			dihedrals_flag = 1;
		} else if (strcmp(keyword,"Impropers")==0) {
			read_impropers(file);
			impropers_flag = 1;
		} else if (strcmp(keyword,"Masses")==0) {
			read_masses(file);
			masses_flag = 1;
		}

		// Read file line by line
		if (fgets(line,MAXLINE,file) == NULL) break;
	}

	fclose(file);

	// Sanity check. All types must be between 1 and N.
	int i,ty;
	bool *bmassty;
	for (i=0;i<natoms;i++) {
		if (atoms[i].type<=0 || atoms[i].type>natom_types)
			print_error("Incorrect atom type in data file\n");
	}
	for (i=0;i<nbonds;i++) {
		if (bonds[i].type<=0 || bonds[i].type>nbond_types)
			print_error("Incorrect bond type in data file\n");
	}
	for (i=0;i<nangles;i++) {
		if (angles[i].type<=0 || angles[i].type>nangle_types)
			print_error("Incorrect angle type in data file\n");
	}
	for (i=0;i<ndihedrals;i++) {
		if (dihedrals[i].type<=0 || dihedrals[i].type>ndihedral_types)
			print_error("Incorrect dihedral type in data file\n");
	}
	for (i=0;i<nimpropers;i++) {
		if (impropers[i].type<=0 || impropers[i].type>nimproper_types)
			print_error("Incorrect improper type in data file\n");
	}
	// Masses have to be defined for all types
	bmassty = new bool[natom_types];
	for (i=0;i<natom_types;i++) bmassty[i]=false;
	for (i=0;i<natom_types;i++) {
		ty = masses[i].id-1;
		if (ty<0 || ty>=natom_types) print_error("Incorrect atom type in masses definition\n");
		if (bmassty[ty]) print_error("Multiple definition of mass for atom type\n");
		bmassty[ty] = true;
	}
	delete [] bmassty;
}

void ReadTopology::read_atoms(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>8) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if ((m!=8 && m!=5) || nread>=natoms) print_error("Incorrect atom format in data file\n");

		atoms[nread].id = atoi(values[0]);
		atoms[nread].mol_id = atoi(values[1]);
		atoms[nread].res_id = atoi(values[2]);
		atoms[nread].type = atoi(values[3]);
		atoms[nread].charge = atof(values[4]);
		if (m>=8) {
			atoms[nread].x = atof(values[5]);
			atoms[nread].y = atof(values[6]);
			atoms[nread].z = atof(values[7]);
			atoms[nread].coord_flag = true;
		}
		nread++;
	}

	if (nread!=natoms) print_error("Incorrect atom format in data file\n");
}

void ReadTopology::read_bonds(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>4) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if (m!=4 || nread>=nbonds) print_error("Incorrect bond format in data file\n");

		bonds[nread].id = atoi(values[0]);
		bonds[nread].type = atoi(values[1]);
		bonds[nread].atom1 = atoi(values[2]);
		bonds[nread].atom2 = atoi(values[3]);
		nread++;
	}

	if (nread!=nbonds) print_error("Incorrect bond format in data file\n");
}

void ReadTopology::read_angles(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>5) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if (m!=5 || nread>=nangles) print_error("Incorrect angle format in data file\n");

		angles[nread].id = atoi(values[0]);
		angles[nread].type = atoi(values[1]);
		angles[nread].atom1 = atoi(values[2]);
		angles[nread].atom2 = atoi(values[3]);
		angles[nread].atom3 = atoi(values[4]);
		nread++;
	}

	if (nread!=nangles) print_error("Incorrect angle format in data file\n");
}

void ReadTopology::read_dihedrals(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>6) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if (m!=6 || nread>=ndihedrals) print_error("Incorrect dihedral format in data file\n");

		dihedrals[nread].id = atoi(values[0]);
		dihedrals[nread].type = atoi(values[1]);
		dihedrals[nread].atom1 = atoi(values[2]);
		dihedrals[nread].atom2 = atoi(values[3]);
		dihedrals[nread].atom3 = atoi(values[4]);
		dihedrals[nread].atom4 = atoi(values[5]);
		nread++;
	}

	if (nread!=ndihedrals) print_error("Incorrect dihedral format in data file\n");
}

void ReadTopology::read_impropers(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>6) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if (m!=6 || nread>=nimpropers) print_error("Incorrect improper format in data file\n");

		impropers[nread].id = atoi(values[0]);
		impropers[nread].type = atoi(values[1]);
		impropers[nread].atom1 = atoi(values[2]);
		impropers[nread].atom2 = atoi(values[3]);
		impropers[nread].atom3 = atoi(values[4]);
		impropers[nread].atom4 = atoi(values[5]);
		nread++;
	}

	if (nread!=nimpropers) print_error("Incorrect improper format in data file\n");
}

void ReadTopology::read_masses(FILE *file)
{
	int nread, m;
	char *line, *ptr;
	char **values = new char*[10];

	line = new char[MAXLINE];

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
		if ((ptr = strchr(line,'#'))!=NULL) *ptr = '\0';
		if (strspn(line," \t\n\r") == strlen(line)) break;

		m = 0;
		values[m] = strtok(line," \t\n\r\f");
		while (values[m]!=NULL) {
			m++;
			if (m>2) break;

			values[m] = strtok(NULL," \t\n\r\f");
		}
		if (m!=2 || nread>=natom_types) print_error("Incorrect mass format in data file\n");

		masses[nread].id = atoi(values[0]);
		masses[nread].mass = atof(values[1]);
		nread++;
	}

	if (nread!=natom_types) print_error("Incorrect mass format in data file\n");
}

char *ReadTopology::trim(char *str)
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

void ReadTopology::print_error(const char *txt)
{
	printf("%s", txt);
	exit(0);
}
