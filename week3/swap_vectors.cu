#include <stdio.h>
#include <iostream>

const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *vec1, float *vec2, int v_size) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx = threadIdx.x + blockIdx.x*block_size;
    // Swap the vector elements - make sure you are not out of range
    if (idx < v_size){
    	float c = vec1[idx];
	vec1[idx] = vec2[idx];
	vec2[idx] = c;
    }
}


int main() {

    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
	
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    std::cout<<"----Before swapping values----"<<std::endl;
    std::cout<<"A vector"<<std::endl;
    for (int i = 0; i < 10; i++){
	    std::cout<< h_A[i]<<" ";
    }
    std::cout<<std::endl<<"B vector"<<std::endl;
    for (int i = 0; i < 10; i++){
    	std::cout<< h_B[i]<<" ";
    }

    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A,d_B,DSIZE);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    // Print and check some elements to make sure swapping was successful
    std::cout<<std::endl<<"----After swapping values----"<<std::endl;
    std::cout<<"A vector"<<std::endl;
    for (int i = 0; i < 10; i++){
	    std::cout<< h_A[i]<<" ";
    }
    std::cout<<std::endl<<"B vector"<<std::endl;
    for (int i = 0; i < 10; i++){
    	std::cout<< h_B[i]<<" ";
    }

    // Free the memory
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B); 

    return 0;
}
