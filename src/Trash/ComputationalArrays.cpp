/*
 * ComputationalArrays.cpp
 *
 *  Created on: 23.08.2012
 *      Author: zhmurov
 */

#include "ComputationalArrays.h"

bool int2_comparator(int2 i, int2 j){
	if(i.x < j.x){
		return true;
	} else
	if(i.x == j.x){
		return i.y < j.y;
	} else {
		return false;
	}
}

ComputationalArrays::ComputationalArrays(ReadTopology *top, ReadParameters *par)
{
	this->top = top;
	this->par = par;
}

ComputationalArrays::~ComputationalArrays()
{
}

void ComputationalArrays::GetBondList(string name, std::vector<int3> *bonds, std::vector<Coeffs> *parameters)
{
	int i,j;
	for (i=0;i<par->nbond_types;i++) {
		if (par->bond_coeffs[i].name == name) {
			parameters->push_back(par->bond_coeffs[i]);
		}
	}

	for (i=0;i<top->nbonds;i++) {
		for (j=0;j!=parameters->size();j++) {
			if (top->bonds[i].type == parameters->at(j).id) {
				int3 b;
				b.x = top->bonds[i].atom1-1;
				b.y = top->bonds[i].atom2-1;
				b.z = j;
				bonds->push_back(b);
				break;
			}
		}
	}
}

void ComputationalArrays::GetExclusionList(std::vector<int2> *list, std::vector<int> *excl_bond_types)
{
	int i, j;
	for (i=0;i<top->nbonds;i++) {
		for (j=0;j!=(*excl_bond_types).size();j++) {
			if( top->bonds[i].type==(*excl_bond_types).at(j) ) {
				int2 p;
				if(top->bonds[i].atom1-1 < top->bonds[i].atom2-1){
					p.x = top->bonds[i].atom1-1;
					p.y = top->bonds[i].atom2-1;
				} else {
					p.x = top->bonds[i].atom2-1;
					p.y = top->bonds[i].atom1-1;
				}
				list->push_back(p);
			}
		}
	}
	std::sort(list->begin(), list->end(), int2_comparator);
}

bool ComputationalArrays::QPairExcluded(int atom1, int atom2, std::vector<int> *excl_bond_types)
{
	int i,j;

	if(atom1 == atom2) return true;

	for (i=0;i<top->nbonds;i++) {
		if ((top->bonds[i].atom1==atom1 && top->bonds[i].atom2==atom2) ||
				(top->bonds[i].atom1==atom2 && top->bonds[i].atom2==atom1)){
				for (j=0;j!=(*excl_bond_types).size();j++) {
					if( top->bonds[i].type==(*excl_bond_types).at(j) ) return true;
				}
			return false;
		}
	}

	return false;
}
