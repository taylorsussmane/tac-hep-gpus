#include<stdio.h>
#include <algorithm>
using namespace std;

#define RADIUS 3
#define BLOCK_SIZE 32
#define N 512
const int DSIZE = N+2*RADIUS;
const int A_val = 1;
const int B_val = 2;

//--------------MATRIX MULTIPLICATION W/ SHARED MEM--------------
// reference: https://github.com/kberkay/Cuda-Matrix-Multiplication/blob/master/matrix_Multiplication.cu
__global__ void matrix_mult(const int *A, const int *B, int *C, int size) {
	// put smaller tiles of matrix into shared mem
	__shared__ int tileA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tileB[BLOCK_SIZE][BLOCK_SIZE];

	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int idy = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	
	int temp = 0;

	// loop through tiles grid
	for (int ntile = 0; ntile < (BLOCK_SIZE+DSIZE-1)/BLOCK_SIZE; ntile++){
		if (ntile*BLOCK_SIZE+ threadIdx.x < DSIZE && idy < DSIZE)
			tileA[threadIdx.y][threadIdx.x] = A[idy*DSIZE+(ntile*BLOCK_SIZE+threadIdx.x)];
		else
			tileA[threadIdx.y][threadIdx.x] = 0;
		if (ntile*BLOCK_SIZE + threadIdx.y < DSIZE && idx < DSIZE)
			tileB[threadIdx.y][threadIdx.x] = B[(ntile*BLOCK_SIZE+threadIdx.y)*DSIZE+idx];
		else
			tileB[threadIdx.y][threadIdx.x] = 0;
		
		__syncthreads();

		// do the matrix multiplication for one tile
		for (int i = 0; i < BLOCK_SIZE; i++){	
			temp+= tileA[threadIdx.y][i]*tileB[i][threadIdx.x];
		}
		__syncthreads();
	}
	// store result
	//C[idy*DSIZE+idx] = temp;
	if(idy < DSIZE && idx < DSIZE)
		C[((blockIdx.y*blockDim.y+threadIdx.y)*DSIZE)+(blockIdx.x*blockDim.x)+threadIdx.x] = temp;
}





/*
    // create thread x index
    // create thread y index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float tempSum = 0;
        for (int i = 0; i < size; i++){
        tempSum += A[idy*size + i]*B[i*size + idx];
        }
        C[idy*size+idx] = tempSum;
    }
}*/

//-----------------2D STENCIL-----------------
__global__ void stencil_2d(int *in, int *out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
	int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex_x = threadIdx.x + RADIUS;
	int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
	int lindex_y = threadIdx.y + RADIUS;

	// Read input elements into shared memory
	int size = N + 2 * RADIUS;
	temp[lindex_x][lindex_y] = in[gindex_y+size*gindex_x];

	if (threadIdx.x < RADIUS) {
		temp[lindex_x-RADIUS][lindex_y] = in[size*(gindex_x - RADIUS)+gindex_y];
		temp[lindex_x+BLOCK_SIZE][lindex_y] = in[size*(gindex_x+BLOCK_SIZE)+gindex_y];
	}

	if (threadIdx.y < RADIUS ) {
		temp[lindex_x][lindex_y-RADIUS] = in[gindex_x*size+(gindex_y-RADIUS)];
		temp[lindex_x][lindex_y+BLOCK_SIZE] = in[gindex_x*size+(gindex_y+BLOCK_SIZE)];
	}

	__syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++){
		result += temp[lindex_x+offset][lindex_y];
		result += temp[lindex_x][lindex_y+offset];
	}

	// Avoid double counting
	result -= temp[lindex_x][lindex_y];

	// Store the result
	out[gindex_y+size*gindex_x] = result;
}


//-----------------2D STENCIL VALUE CHECK-----------------
int stencil_error_check(const int *in, const int *out, const int val){
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

//-----------------MATRIX MULT VALUE CHECK-----------------
int mult_error_check(const int *A, const int *B, const int *C){
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



void fill_ints(int *x, int n, int val) {
   fill_n(x, n, val);
}


//-----------------ERROR CHECKING-----------------
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

int main(void){

	int *h_a, *h_b, *h_stenciled_a, *h_stenciled_b, *h_c; //host copies
	
	int size = (N+2*RADIUS)*(N+2*RADIUS)*sizeof(int);

	// create streams
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	
	// allocate space for device copies with managed memory
	cudaMallocManaged((void **)&h_a, size);
	cudaStreamAttachMemAsync(stream1, h_a, size);
	cudaMallocManaged((void **)&h_b, size);
	cudaStreamAttachMemAsync(stream2, h_b, size);
	cudaMallocManaged((void **)&h_c, size);
	cudaMallocManaged((void **)&h_stenciled_a, size);
	cudaStreamAttachMemAsync(stream1, h_stenciled_a, size, cudaMemAttachGlobal);
	cudaMallocManaged((void **)&h_stenciled_b, size);
	cudaStreamAttachMemAsync(stream2, h_stenciled_b, size, cudaMemAttachGlobal);
	cudaCheckErrors("Error when allocating memory.");
	
	// initialize values
	fill_ints(h_a, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
	fill_ints(h_b, (N + 2*RADIUS)*(N + 2*RADIUS), B_val);
	fill_ints(h_stenciled_a, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
	fill_ints(h_stenciled_b, (N + 2*RADIUS)*(N + 2*RADIUS), B_val);
	fill_ints(h_c, (N + 2*RADIUS)*(N + 2*RADIUS), 0);

	// create grid and block for stencil kernel
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	// launch stencil kernels in streams 1 and 2 
	stencil_2d<<<grid,block,0,stream1>>>(h_a + RADIUS*(N + 2*RADIUS) + RADIUS , h_stenciled_a + RADIUS*(N + 2*RADIUS) + RADIUS);
	stencil_2d<<<grid,block,0,stream2>>>(h_b + RADIUS*(N + 2*RADIUS) + RADIUS , h_stenciled_b + RADIUS*(N + 2*RADIUS) + RADIUS);
	cudaCheckErrors("Error when running stencil kernels.");
	
	// synchronize before accessing data on host
	cudaDeviceSynchronize();

	// synchronize streams
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	// launch matrix mult
	int m_gridSize = DSIZE/BLOCK_SIZE;
	dim3 m_grid(m_gridSize, m_gridSize);
	dim3 m_block(BLOCK_SIZE, BLOCK_SIZE);
	matrix_mult<<<m_grid,m_block>>>(h_stenciled_a, h_stenciled_b, h_c, DSIZE);
	cudaCheckErrors("Error when running multiplication kernel.");
	
	// check results
	stencil_error_check(h_a, h_stenciled_a, A_val);
	stencil_error_check(h_b, h_stenciled_b, B_val);
	mult_error_check(h_stenciled_a, h_stenciled_b, h_c);

	// free memory
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);
	cudaFree(h_stenciled_a);
	cudaFree(h_stenciled_b);

	// destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

}
