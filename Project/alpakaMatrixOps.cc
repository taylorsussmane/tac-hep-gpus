#include <stdio.h>
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include "config.h"
#include "WorkDiv.hpp"

using namespace alpaka;

#define RADIUS 3
#define N 512
#define BLOCK_SIZE 32
#define A_val 1
#define B_val 2
#define DSIZE (N+2*RADIUS)

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

//--------------MATRIX MULTIPLICATION--------------
struct matrix_mult{
	template <typename TAcc, typename T>
	ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ in_A,
                                  T const* __restrict__ in_B,
                                  T* __restrict__ out,
                                  Vec2D size) const {	
			auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
			int threadIdxX = globalThreadIdx[0];
			int threadIdxY = globalThreadIdx[1];
			auto blocksize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
			int blockdimX = blocksize[0];
			int blockdimY = blocksize[1];
			auto blockId = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
			int blockIdX = blockId[0];
			int blockIdY = blockId[1];
			
			int idx = blockIdX * blockdimX + threadIdxX;
			int idy = blockIdY * blockdimY + threadIdxY;
			
			if ((idx < size) && (idy < size)) {
				float tempSum = 0;
				for (int i = 0; i < size; i++){
				tempSum += A[idy*size + i]*B[i*size + idx];
				}
				C[idy*size+idx] = tempSum;	
			}
	}
};

//-----------------2D STENCIL-----------------
struct stencil_2d{
	template <typename TAcc, typename T>
	ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ in_A,
                                  T const* __restrict__ in_B,
                                  T* __restrict__ out,
                                  Vec2D size) const {	
			auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
			int threadIdxX = globalThreadIdx[0];
			int threadIdxY = globalThreadIdx[1];
			auto blocksize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
			int blockdimX = blocksize[0];
			int blockdimY = blocksize[1];
			auto blockId = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
			int blockIdX = blockId[0];
			int blockIdY = blockId[1];
		
			int gindex_x = threadIdxX + blockIdX * blockdimX;
			int lindex_x = threadIdxX + RADIUS;
			int gindex_y = threadIdxY + blockIdY * blockdimY;
			int lindex_y = threadIdxY + RADIUS;
			auto& temp = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);
			
			// Read inputs into shared memory
			temp[lindex_x][lindex_y] = in[gindex_x*DSIZE + gindex_y];

			if (threadIdx.x < RADIUS) {
				temp[lindex_x - RADIUS][lindex_y] = in[(gindex_x - RADIUS)*size + gindex_y];
				temp[lindex_x + BLOCK_SIZE][lindex_y] = in[(gindex_x + BLOCK_SIZE)*size + gindex_y];
			}

			if (threadIdx.y < RADIUS ) {
				temp[lindex_x][lindex_y - RADIUS] = in[gindex_x*size + (gindex_y - RADIUS)];
				temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_x*size + (gindex_y + BLOCK_SIZE)];
			}

			syncBlockThreads(acc);
			
			// apply stencil
			int result = 0;
			for (int offset = -RADIUS; offset <= RADIUS; offset++){
				result += temp[lindex_x + offset][lindex_y];
				result += temp[lindex_x][lindex_y + offset];
			}
			//avoid double counting
			result -= temp[lindex_x][lindex_y];
			
			// Store the result
			out[gindex_x*size + gindex_y] = result;
	}
};

//-----------------2D STENCIL VALUE CHECK-----------------
int stencil_checker(const int *out, int value){
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
		for (int j = 0; j < N + 2 * RADIUS; ++j) {

			if (i < RADIUS || i >= N + RADIUS) {
				if (out[j+i*(N + 2 * RADIUS)] != value) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], value);
					return -1;
				}
			}
			else if (j < RADIUS || j >= N + RADIUS) {
				if (out[j+i*(N + 2 * RADIUS)] != value) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], value);
					return -1;
				}
			}		 
			else {
				if (out[j+i*(N + 2 * RADIUS)] != value + value * 4 * RADIUS) {
					printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, out[j+i*(N + 2 * RADIUS)], value + value * 4 * RADIUS);
					return -1;
				}
			}
		}
	}
	printf("Stencil success!")
    return 0;
}

//-----------------MATRIX MULT VALUE CHECK-----------------
int mult_checker(const int *A, const int *B, const int *C){
	int A_stenc_val = A_val + A_val*4*RADIUS;
	int B_stenc_val = B_val + B_val*4*RADIUS;
	int DSIZE = N + 2*RADIUS;
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
		for (int j = 0; j < N + 2 * RADIUS; ++j) {
			if ((i < RADIUS || i >= N + RADIUS) && (j < RADIUS || j >= N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*DSIZE){
					printf("1 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*DSIZE);
					return -1;
				}
			}
			else if ((i < RADIUS || i >= N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_val*B_stenc_val*N){
					printf("2 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_val*B_stenc_val*N);
					return -1;
				}
			}
			else if ((i >= RADIUS && i < N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*N){
					printf("3 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*N);
					return -1;
				}
			}
			else{
				if (C[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_val*N){
					printf("4 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_val*N);
					return -1;
				}
			}
		}
	}
	printf("Matrix Multplication Success!")
	return 0;
}

void fill_ints(int *x, int size, int n) {
   // Store the result
   // https://en.cppreference.com/w/cpp/algorithm/fill_n
   fill_n(x, size, n);
}

int main(void){

	//require at least one device
    std::size_t n = getDevCount<Platform>();
    if (n==0) {
        exit(EXIT_FAILURE);
    }

	// use the single host device
  	HostPlatform host_platform;
  	Host host = alpaka::getDevByIdx(host_platform, 0u);
  	std::cout << "Host:   " << alpaka::getName(host) << '\n';

  	// use the first device
  	Device device = alpaka::getDevByIdx(platform, 0u);
  	std::cout << "Device: " << alpaka::getName(device) << '\n';	

}
