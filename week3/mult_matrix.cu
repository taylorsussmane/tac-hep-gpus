#include <stdio.h>
#include <time.h>

const int DSIZE = 256;
const float A_val = 3.0f;
const float B_val = 2.0f;

// error checking macro
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

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
	for (int i = 0; i<size; i++){
		for (int j = 0; j<size; j++){
			float tempSum = 0;

			for (int k = 0; k<size; k++){
				tempSum += A[i*size+k]+B[k*size+j];
			}

			C[i*size+j] = tempSum;
		}
	} 
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {

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

int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // These are used for timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t2sum=0.0;
    double t3sum=0.0;

    // start timing
    t0 = clock();

    // N*N matrices defined in 1 dimention
    // If you prefer to do this in 2-dimentions cupdate accordingly
    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data from host to device
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
	cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
	cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));

	cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  

    // Launch kernel
    // Specify the block and grid dimentions 
    dim3 block(DSIZE, DSIZE);
    dim3 grid(1, 1);
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. GPU compute took %f seconds\n", t2sum);

    // Excecute and time the cpu matrix multiplication function
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);

    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Done. CPU compute took %f seconds\n", t3sum);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); 
    
    return 0;

}
