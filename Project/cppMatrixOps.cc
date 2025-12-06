#include <iostream>
#include<stdio.h>
#include<cstdlib>

using namespace std

const int DSIZE = 512;
const int RADIUS = 3;
const int A_val = 1;
const int B_val = 2;

void stencil_2d(const int in[][DSIZE], int out[][DSIZE]){
	for(int x_i = 0; x_i < DSIZE; x_i++){
		for (int y_i = 0; y_i < DSIZE; y_i++){
			int n = in[x_i][y_i];
			if (x_i < RADIUS || x_i + RADIUS >= DSIZE){
				out[x_i][y_i] = n;
			}
			else if (y_i < RADIUS || y_i + RADIUS >= DSIZE){
				out[x_i][y_i] = n;
			}
		}
	}

}
	//create 2d matrices A and B of size DSIZE = 512 and fill them with random values
}

function2{
	// 2d stencil operation on matrices of radius >2
}

function3{
	// matrix multiplication 
}

function4{
	// check results
}

int main{

//stuff

}
