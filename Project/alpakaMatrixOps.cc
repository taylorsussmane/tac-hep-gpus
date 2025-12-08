#include <stdio.h>
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include "config.h"
#include "WorkDiv.hpp"

using namespace alpaka;
using namespace std;

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
                                  T const* __restrict__ A,
                                  T const* __restrict__ B,
                                  T* __restrict__ C,
                                  Vec2D size) const {	
			auto globalThreadIdx = getIdx<Grid, Threads>(acc);
			int threadIdxX = globalThreadIdx[0];
			int threadIdxY = globalThreadIdx[1];
			auto blocksize = getWorkDiv<Block, Threads>(acc);
			int blockdimX = blocksize[0];
			int blockdimY = blocksize[1];
			auto blockId = getWorkDiv<Grid, Blocks>(acc);
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
                                  T const* __restrict__ in,
                                  T* __restrict__ out,
                                  Vec2D size) const {	
			auto globalThreadIdx = getIdx<Grid, Threads>(acc);
			int threadIdxX = globalThreadIdx[0];
			int threadIdxY = globalThreadIdx[1];
			auto blocksize = getWorkDiv<Block, Threads>(acc);
			int blockdimX = blocksize[0];
			int blockdimY = blocksize[1];
			auto blockId = getWorkDiv<Grid, Blocks>(acc);
			int blockIdX = blockId[0];
			int blockIdY = blockId[1];
		
			int gindex_x = threadIdxX + blockIdX * blockdimX;
			int lindex_x = threadIdxX + RADIUS;
			int gindex_y = threadIdxY + blockIdY * blockdimY;
			int lindex_y = threadIdxY + RADIUS;
			auto& temp = declareSharedVar<std::uint32_t, __COUNTER__>(acc);
			
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
int stencil_error_check(const int *out, int value){
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
int mult_error_check(const int *A, const int *B, const int *C){
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
  	Host host = getDevByIdx(host_platform, 0u);
  	std::cout << "Host:   " << getName(host) << '\n';

  	// use the first device
  	Device device = getDevByIdx(platform, 0u);
  	std::cout << "Device: " << getName(device) << '\n';	

	// 2D and linearized buffer size
    constexpr Vec2D ndsize = {DSIZE,DSIZE};
    constexpr size_t size = ndsize.prod();

	// allocate input and output host buffers
    auto h_a = allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{size});
    auto h_stenciled_a = allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{size});
    auto h_b = allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{size});
    auto h_stenciled_b = allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{size});
    auto h_c = allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{size});

	// fill input buffers
	for (size_t i = 0; i < m_size; i++) {
        in_A_h[i] = A_val;
        out_A_h[i] = A_val;
        in_B_h[i] = B_val;
        out_B_h[i] = B_val;
        C_h[i] = 0;
    }
	
	// create queue and allocate buffers on device
	auto queue = Queue{device};
	auto h_a = allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
	auto h_stenciled_a = allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
	auto h_b = allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
	auto h_stenciled_b = allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
	auto h_c = allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});

	// copy to device
	memcpy(queue, d_a, h_a);
	memcpy(queue, d_stenciled_a, h_stenciled_a);
	memcpy(queue, d_b, h_b);
	memcpy(queue, d_stenciled_b, h_stenciled_b);

	// fill the output buffer with zeros; the size is known from the buffer objects
	memset(queue, d_c, 0x00);	

	// launch kernels
	int gridsize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	auto div = makeWorkDiv<Acc2D>({gridsize, gridsize}, {BLOCK_SIZE, BLOCK_SIZE});
	std::cout << "Testing stencil_2d and matrix_mult kernels with vector indices with a grid of "
		<< getWorkDiv<Grid, Blocks>(div) << " blocks x "
		<< getWorkDiv<Block, Threads>(div) << " threads x "
		<< getWorkDiv<Thread, Elems>(div) << " elements...\n";
	exec<Acc2D>(queue, div, stencil_2d{}, d_a.data(), d_stenciled_a.data(), ndsize);
	exec<Acc2D>(queue, div, stencil_2d{}, d_b.data(), d_stenciled_b.data(), ndsize);
	exec<Acc2D>(queue, div, matrix_mult{}, d_stenciled_a.data(), d_stenciled_b.data(), d_c.data(), ndsize);

	// copy results back to host
	memcpy(queue, h_c, d_c);
	memcpy(queue, h_stenciled_a, d_stenciled_a);
	memcpy(queue, h_stenciled_b, d_stenciled_b);

	// wait for all operations to complete
	wait(queue);

	// check the results
	stencil_error_check(h_stenciled_a, A_val);
	stencil_error_check(h_stenciled_b, B_val);
	mult_error_check(h_stenciled_a, h_stenciled_b, h_c);
}
