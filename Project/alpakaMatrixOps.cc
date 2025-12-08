#include <stdio.h>
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include "config.h"
#include "WorkDiv.hpp"

#define RADIUS 3
#define N 512
#define BLOCK_SIZE 32
#define A_val 1
#define B_val 2
#define DSIZE (N+2*RADIUS)


//--------------MATRIX MULTIPLICATION--------------
struct matrix_mult{
	template <typename TAcc, typename T>
	ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ A,
                                  T const* __restrict__ B,
                                  T* __restrict__ C,
								  Vec2D size) const {	
			auto globalThreadIdx = getIdx<alpaka::Grid, alpaka::Threads>(acc);
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
			
			if ((idx < DSIZE) && (idy < DSIZE)) {
				int tempSum = 0;
				for (int i = 0; i < DSIZE; i++){
				tempSum += A[idy*DSIZE + i]*B[i*DSIZE + idx];
				}
				C[idy*DSIZE+idx] = tempSum;	
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
			int gindex_y = threadIdxY + blockIdY * blockdimY;

			syncBlockThreads(acc);
			
			// apply stencil
			int result = 0;
			for (int offset = -RADIUS; offset <= RADIUS; offset++){
				result += in[gindex_y+(gindex_x+offset)*DSIZE];
				result += in[gindex_y+offset+gindex_x*DSIZE];
			}
			//avoid double counting
			result -= in[gindex_y+gindex_x*DSIZE];
			
			// Store the result
			out[gindex_y+DSIZE*gindex_x] = result;
	}
};


int main(void){
	Platform platform;

	//require at least one device
    std::size_t n = alpaka::getDevCount(platform);
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

	// 2D and linearized buffer size
    constexpr Vec2D ndsize = {DSIZE,DSIZE};
    constexpr size_t size = ndsize.prod();

	// allocate input and output host buffers
    auto h_a = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{size});
    auto h_stenciled_a = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{size});
    auto h_b = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{size});
    auto h_stenciled_b = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{size});
    auto h_c = alpaka::allocMappedBuf<int, uint32_t>(host, platform, Vec1D{size});

	// fill input buffers
	for (size_t i = 0; i < size; i++) {
        h_a[i] = A_val;
        h_stenciled_a[i] = A_val;
        h_b[i] = B_val;
        h_stenciled_b[i] = B_val;
        h_c[i] = 0;
    }
	
	// create queue and allocate buffers on device
	auto queue = Queue{device};
	auto d_a = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{size});
	auto d_stenciled_a = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{size});
	auto d_b = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{size});
	auto d_stenciled_b = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{size});
	auto d_c = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{size});

	// copy to device
	alpaka::memcpy(queue, d_a, h_a);
	alpaka::memcpy(queue, d_stenciled_a, h_stenciled_a);
	alpaka::memcpy(queue, d_b, h_b);
	alpaka::memcpy(queue, d_stenciled_b, h_stenciled_b);

	// fill the output buffer with zeros; the size is known from the buffer objects
	alpaka::memset(queue, d_c, 0x00);	

	// launch kernels
	int gridsize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	auto div = makeWorkDiv<Acc2D>({gridsize, gridsize}, {BLOCK_SIZE, BLOCK_SIZE});
	std::cout << "Testing stencil_2d and matrix_mult kernels with vector indices with a grid of "
		<< alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
		<< alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
		<< alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
	alpaka::exec<Acc2D>(queue, div, stencil_2d{}, d_a.data(), d_stenciled_a.data(), ndsize);
	alpaka::exec<Acc2D>(queue, div, stencil_2d{}, d_b.data(), d_stenciled_b.data(), ndsize);
	alpaka::exec<Acc2D>(queue, div, matrix_mult{}, d_stenciled_a.data(), d_stenciled_b.data(), d_c.data(), ndsize);

	// copy results back to host
	alpaka::memcpy(queue, h_c, d_c);
	alpaka::memcpy(queue, h_stenciled_a, d_stenciled_a);
	alpaka::memcpy(queue, h_stenciled_b, d_stenciled_b);

	// wait for all operations to complete
	alpaka::wait(queue);

	// check the results
	
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
            for (int j = 0; j < N + 2 * RADIUS; ++j) {

                if (i < RADIUS || i >= N + RADIUS) {
                    if (h_stenciled_a[j+i*(N + 2 * RADIUS)] != A_val) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_a[j+i*(N + 2 * RADIUS)], A_val);
                        return -1;
                    }
                }
                else if (j < RADIUS || j >= N + RADIUS) {
                    if (h_stenciled_a[j+i*(N + 2 * RADIUS)] != A_val) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_a[j+i*(N + 2 * RADIUS)], A_val);
                        return -1;
                    }
                }
                else {
                    if (h_stenciled_a[j+i*(N + 2 * RADIUS)] != A_val + A_val*4 * RADIUS) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_a[j+i*(N + 2 * RADIUS)], A_val + A_val*4*RADIUS);
                        return -1;
                    }
                }
            }
        }
    printf("A Stencil Success!\n");
    return 0;

	for (int i = 0; i < N + 2 * RADIUS; ++i) {
            for (int j = 0; j < N + 2 * RADIUS; ++j) {

                if (i < RADIUS || i >= N + RADIUS) {
                    if (h_stenciled_b[j+i*(N + 2 * RADIUS)] != B_val) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_b[j+i*(N + 2 * RADIUS)], B_val);
                        return -1;
                    }
                }
                else if (j < RADIUS || j >= N + RADIUS) {
                    if (h_stenciled_b[j+i*(N + 2 * RADIUS)] != B_val) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_b[j+i*(N + 2 * RADIUS)], B_val);
                        return -1;
                    }
                }
                else {
                    if (h_stenciled_b[j+i*(N + 2 * RADIUS)] != B_val + B_val*4 * RADIUS) {
                        printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_stenciled_b[j+i*(N + 2 * RADIUS)], B_val + B_val*4*RADIUS);
                        return -1;
                    }
                }
            }
        }
    printf("B Stencil Success!\n");
    return 0;
	
	int A_stenc_val = A_val + A_val*4*RADIUS;
	int B_stenc_val = B_val + B_val*4*RADIUS;
	for (int i = 0; i < N + 2 * RADIUS; ++i) {
		for (int j = 0; j < N + 2 * RADIUS; ++j) {
			if ((i < RADIUS || i >= N + RADIUS) && (j < RADIUS || j >= N + RADIUS)){
				if (h_c[j+i*DSIZE] != A_val*B_val*DSIZE){
					printf("1 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_c[j+i*DSIZE], A_val*B_val*DSIZE);
					return -1;
				}
			}
			else if ((i < RADIUS || i >= N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (h_c[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_val*B_stenc_val*N){
					printf("2 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_c[j+i*DSIZE], A_val*B_val*2*RADIUS + A_val*B_stenc_val*N);
					return -1;
				}
			}
			else if ((i >= RADIUS && i < N + RADIUS) && (j >= RADIUS && j < N + RADIUS)){
				if (h_c[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*N){
					printf("3 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_c[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_stenc_val*N);
					return -1;
				}
			}
			else{
				if (h_c[j+i*DSIZE] != A_val*B_val*2*RADIUS + A_stenc_val*B_val*N){
					printf("4 Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, h_c[j+i*DSIZE], A_val*B_val*2*RADIUS + A_stenc_val*B_val*N);
					return -1;
				}
			}
		}
	}
	printf("Matrix Multplication Success!");
	return 0;
}
