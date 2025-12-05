#include<stdio.h>
#include <algorithm>
using namespace std;

#define N 512
#define RADIUS 2
#define BLOCK_SIZE 32
const int DSIZE = N+2*RADIUS;
//#define N (DSIZE-2*RADIUS)
//#define A_val 1
//#define B_val 2
const int A_val = 1;
const int B_val = 2;

//--------------MATRIX MULTIPLICATION--------------
__global__ void matrix_mult(const int *A, const int *B, int *C, int size) {

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
}

//-----------------2D STENCIL-----------------
__global__ void stencil_2d(int *in, int *out) {

    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
    int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Read input elements into shared memory
    int size = N + 2 * RADIUS;

    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += in[gindex_y+(gindex_x+offset)*size];
        result += in[gindex_y+offset+gindex_x*size];
    }

    // In the loop, this index is double counted, so this fixes that.
    result -= in[gindex_y+gindex_x*size];

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
	printf("Success!\n");
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
	int *d_a, *d_b, *d_stenciled_a, *d_stenciled_b, *d_c; //device copies
	
	// allocate space for the host copies and fill them will random ints
	int size = (N+2*RADIUS)*(N+2*RADIUS)*sizeof(int);
	h_a = (int *)malloc(size); fill_ints(h_a, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
	h_b = (int *)malloc(size); fill_ints(h_b, (N + 2*RADIUS)*(N + 2*RADIUS), B_val);
	h_stenciled_a = (int *)malloc(size); fill_ints(h_stenciled_a, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
	h_stenciled_b = (int *)malloc(size); fill_ints(h_stenciled_b, (N + 2*RADIUS)*(N + 2*RADIUS), B_val); 
	h_c = (int *)malloc(size); fill_ints(h_c, (N + 2*RADIUS)*(N + 2*RADIUS), 0);

	// allocate space for device copies
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	cudaMalloc((void **)&d_stenciled_a, size);
	cudaMalloc((void **)&d_stenciled_b, size);
	cudaCheckErrors("Error when allocating memory.");
	
	// copy to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stenciled_a, h_stenciled_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stenciled_b, h_stenciled_b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
	cudaCheckErrors("Error when copying to device.");

	// create grid and block for stencil kernel
	int s_gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 s_grid(s_gridSize, s_gridSize);
	dim3 s_block(BLOCK_SIZE, BLOCK_SIZE);
	
	// launch stencil kernels on a and b and then launch matrix multiplication 
	stencil_2d<<<s_grid,s_block>>>(d_a + RADIUS*(N + 2*RADIUS) + RADIUS , d_stenciled_a + RADIUS*(N + 2*RADIUS) + RADIUS);
	stencil_2d<<<s_grid,s_block>>>(d_b + RADIUS*(N + 2*RADIUS) + RADIUS , d_stenciled_b + RADIUS*(N + 2*RADIUS) + RADIUS);
	cudaCheckErrors("Error when running stencil kernels.");

	//create grid and block for matrix kernel
	//dim3 m_block(DSIZE, DSIZE);
	//dim3 m_grid(1, 1);
	matrix_mult<<<s_grid,s_block>>>(d_stenciled_a, d_stenciled_b, d_c, DSIZE);
	cudaCheckErrors("Error when running multiplication kernel.");
	
	// copy back to host
	cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_stenciled_a, d_stenciled_a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_stenciled_b, d_stenciled_b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaCheckErrors("Error when copying back to host.");
	
	// check results
	stencil_error_check(h_a, h_stenciled_a, A_val);
	stencil_error_check(h_b, h_stenciled_b, B_val);

	// free memory
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_stenciled_a);
	free(h_stenciled_b);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_stenciled_a);
	cudaFree(d_stenciled_b);

}
