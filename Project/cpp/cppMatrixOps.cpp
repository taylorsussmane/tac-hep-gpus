#include <iostream>
#include<stdio.h>
#include<cstdlib>

using namespace std;

const int DSIZE = 512;
const int N = 512;
const int RADIUS = 3;
const int A_val = 1;
const int B_val = 2;

//-----------------2D STENCIL-----------------
void stencil_2d(const int in[][DSIZE], int (&out)[][DSIZE]){
	for(int x_i = 0; x_i < DSIZE; x_i++){
		for (int y_i = 0; y_i < DSIZE; y_i++){
			int n = in[x_i][y_i];
			if (x_i < RADIUS || x_i + RADIUS >= DSIZE){
				out[x_i][y_i] = n;
			}
			else if (y_i < RADIUS || y_i + RADIUS >= DSIZE){
				out[x_i][y_i] = n;
			}
			else {
				int temp = 0;
				for (int offset = -RADIUS; offset <= RADIUS; offset++){
					temp += in[x_i+offset][y_i];
					temp += in[x_i][y_i+offset];
				}
			temp -= n; // double counting
			out[x_i][y_i] = temp;
			}
		}
	}
}

//--------------MATRIX MULTIPLICATION--------------
void matrix_mult(const int A[][DSIZE], const int B[][DSIZE], int (&C)[][DSIZE]){
	for (int i = 0; i < DSIZE; i++){
		for (int j = 0; j < DSIZE; j++){
			int sum = 0;
			for (int k = 0; k < DSIZE; k++){
				sum += A[i][k]*B[k][j];
			}
			C[i][j] = sum;
		}
	}
}

//-----------------2D STENCIL VALUE CHECK-----------------
int stencil_error_check(const int A_out[][DSIZE], const int B_out[][DSIZE]){
	for (int i = 0; i < DSIZE; ++i) {
		for (int j = 0; j < DSIZE; ++j) {
			if (i < RADIUS || i + RADIUS >= DSIZE) {
				if (A_out[i][j] != A_val) {
					printf("A: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, A_out[i][j], A_val);
					return -1;
				}
				if (B_out[i][j] != B_val) {
                    printf("B: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, B_out[i][j], B_val);
                    return -1;
                }
			}
			else if (j < RADIUS || j + RADIUS >= DSIZE) {
				if (A_out[i][j] != A_val) {
                    printf("A: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, A_out[i][j], A_val);
                    return -1;
                }
                if (B_out[i][j] != B_val) {
                    printf("B: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, B_out[i][j], B_val);
                    return -1;
                }
			
			}
			else {
				if (A_out[i][j] != A_val + A_val*4*RADIUS) {
					printf("A: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, A_out[i][j], A_val + A_val*4*RADIUS);
					return -1;
				}
				 if (B_out[i][j] != B_val + B_val*4*RADIUS) {
                    printf("B: mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, B_out[i][j], B_val + B_val*4*RADIUS);
                    return -1;
                    }
			}
		}
	}
	printf("Stencil Success!\n");
	return 0;
}


/*
int stencil_error_check(const int in[][DSIZE], const int out[][DSIZE], const int *val){
		for (int i = 0; i < N + 2 * RADIUS; ++i) {
			for (int j = 0; j < N + 2 * RADIUS; ++j) {

				if (i < RADIUS || i >= N + RADIUS) {
					if (out[j+i*(N + 2 * RADIUS)] != val) {
						printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], val);
						return -1;
					}
				}
				else if (j < RADIUS || j >= N + RADIUS) {
					if (out[j+i*(N + 2 * RADIUS)] != val) {
						printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], val);
						return -1;
					}
				}
				else {
					if (out[j+i*(N + 2 * RADIUS)] != val + val*4 * RADIUS) {
						printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], val + val*4*RADIUS);
						return -1;
					}
				}
			}
		}
	printf("Stencil Success!\n");
	return 0;
}
*/

//-----------------MATRIX MULT VALUE CHECK-----------------
int mult_error_check(const int A[][DSIZE], const int B[][DSIZE], const int C[][DSIZE]){
	int A_stenc_val = A_val + A_val*4*RADIUS;
	int B_stenc_val = B_val + B_val*4*RADIUS;
	int DSIZE_mid = DSIZE - 2*RADIUS;
	for (int i = 0; i < DSIZE; ++i) {
		for (int j = 0; j < DSIZE; ++j) {
			if ((i < RADIUS || i + RADIUS >= DSIZE) && (j < RADIUS || j + RADIUS >= DSIZE)){
				if (C[i][j] != A_val*B_val*DSIZE){
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i][j], A_val*B_val*DSIZE);
					return -1;
				}
			}
			else if ((i < RADIUS || i + RADIUS >= DSIZE) && (j >= RADIUS && j + RADIUS < DSIZE)){
				if (C[i][j] != A_val*B_val*2*RADIUS + A_val*B_stenc_val*DSIZE_mid){
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i][j], A_val*B_val*2*RADIUS + A_val*B_stenc_val*DSIZE_mid);
					return -1;
				}
			}
			else if ((i >= RADIUS && i + RADIUS < DSIZE) && (j >= RADIUS && j + RADIUS < DSIZE)){
				if (C[i][j] != A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*DSIZE_mid){
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i][j], A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*DSIZE_mid);
					return -1;
				}
			}
			else{
				if (C[i][j] != A_val*B_val*2*RADIUS + A_stenc_val*B_val*DSIZE_mid){
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[i][j], A_val*B_val*2*RADIUS + A_stenc_val*B_val*DSIZE_mid);
					return -1;
				}
			}
		}
	}
	printf("Matrix Multiplication Success!\n");
	return 0;
}

/*
int mult_error_check(const int A[][DSIZE], const int B[][DSIZE], const int C[][DSIZE]){
	int A_stenc_val = A_val + A_val*4*RADIUS;
	int B_stenc_val = B_val + B_val*4*RADIUS;
	int DSIZE_mid = DSIZE - 2*RADIUS;
	for (int i = 0; i < DSIZE; ++i) {
		for (int j = 0; j < DSIZE; ++j) {
			if ((i < RADIUS || i >= N + RADIUS) && (j < RADIUS || j >= N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*DSIZE){
					printf("1 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*DSIZE);
					return -1;
				}
			}
			else if ((i < RADIUS || i >= N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_val*B_stenc_val*DSIZE_mid){
					printf("2 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_val*B_stenc_val*DSIZE_mid);
					return -1;
				}
			}
			else if ((i >= RADIUS && i < N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*DSIZE_mid){
					printf("3 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*DSIZE_mid);
					return -1;
				}
			}
			else{
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_val*DSIZE_mid){
					printf("4 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_val*DSIZE_mid);
					return -1;
				}
			}
		}
	}
	printf("Matrix Multiplication Success!\n");
	return 0;
}
*/

int main(){

	int A[DSIZE][DSIZE];
	int B[DSIZE][DSIZE];
	int stenciledA[DSIZE][DSIZE];
	int stenciledB[DSIZE][DSIZE];
	int C[DSIZE][DSIZE];
	
	for(int i = 0; i < DSIZE; i++){
		for (int j = 0; j < DSIZE; j++){
			A[i][j] = A_val;
			B[i][j] = B_val;
			stenciledA[i][j] = 0;
			stenciledB[i][j] = 0;
			C[i][j] = 0;
		}
	}

	stencil_2d(A, stenciledA);
	stencil_2d(B, stenciledB);

	stencil_error_check(stenciledA, stenciledB);

	matrix_mult(stenciledA, stenciledB, C);

	mult_error_check(stenciledA, stenciledB, C);

}
