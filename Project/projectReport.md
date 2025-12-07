# **Final Project**

## C++ and CPU profiling 
- Compiled with 'g++ cppMatrixOps.cpp -o cppMatrixOps'
- Vtune commands:
	1. 'source /opt/intel/oneapi/setvars.sh'
	2. 'vtune -collect hotspots -quiet ./cppMatrixOps'
	3. 'vtune -report summary -result-dir r000hs -format csv -report-output summary.csv'
	4. 'vtune -report hotspots -result-dir r000hs -format csv -report-output hotspots.csv'
- 'matrix\_mult' command is most compute intensive, taking up 94.4% of compute time
- 'stencil\_2d' command is 2nd most compute intensive, taking up 5.6% of compute time
- Successful output:
'''
Stencil Success!
Matrix Multiplication Success!
'''

## Porting to CUDA
- First, 'ssh g38nXX # XX:01-16'
###EXPLICIT MEMORY COPIES
- Compiled with 'nvcc explicitCudaMatrixOps.cu -o explicitCudaMatrixOps'
- Run with './explicitCudaMatrixOps'
- Successful output: 
'''
Stencil Success!
Stencil Success!
Matrix Multiplication Success!
'''
- Run nsys profiler 'nsys profile --stats=true ./explicitCudaMatrixOps'
	- Unable to open GUI because Mac doesn't like when you try to open stuff like that in an ssh
		Put report in txt file 'nsys stats report1.nsys-rep > explicitCudaReport.txt'
		Sometimes need to add '-force-export=true' arg
- 'cudaMalloc' took 98.4% of total time (279,715,106 ns)
- For kernel time:
	'matrix\_mult' took 77.4% of time (745,747 ns)
	'stencil\_2d' took 22.6% of time (217,820 ns)

###MANAGED MEMORY
- Compiled with 'nvcc managedCudaMatrixOps.cu -o managedCudaMatrixOps'
- Ran with './managedCudaMatrixOps'
- Successful output:
'''
Stencil Success!
Stencil Success!
Matrix Multiplication Success!
'''
- Run nsys profiler 'nsys profile --stats=true ./managedCudaMatrixOps'
	- Put report in txt file 'nsys stats report2.nsys-rep > managedCudaReport.txt'
- 'cudaMallocManaged' took 96.6% of total time (268,604,597 ns)
- Kernel time:
	'stencil_2d' took 76.8% of time (5,637,211 ns)
	'matrix_mult' took 23.2% of time (1,701,186 ns)

## Optimizing performance in CUDA
- Implemented non-default stream by doing the stenciling in 2 seperate streams and then merging those streams back together before doing the matrix multiplication.
- Implemented shared memory
	- In stencil, put 'BLOCK\_SIZE+2\*RADIUS' elements in shared memory
	- In matrix multiplication, put square tiles of size 'BLOCK\_SIZE' into shared memory
- Compile with 'nvcc optimizedCudaMatrixOps.cu -o optimizedCudaMatrixOps'
- Run with './optimizedCudaMatrixOps'
- Successful output:
'''
Stencil Success!
Stencil Success!
Matrix Multiplication Success!
'''
- Run nsys profiler 'nsys profile --stats=true ./optimizedCudaMatrixOps'
	- Put report in txt file 'nsys stats report3.nsys-rep > optimizedCudaReport.txt'
- 'cudaStreamCreate' took 96.5% of total time (254,999,751 ns)
- Kernal time:
	'stencil_2d' took 66.8% of time (4,015,963 ns)
	'matrix_mult' took 33.2% of time (1,995,581 ns)

## Making use of Alpaka
- Set up:
	- 'git clone https://github.com/alpaka-group/alpaka.git -b 2.0.0 ${HOME}/public/alpaka'
	- 'git clone https://github.com/kokkos/mdspan.git ${HOME}/public/mdspan'
	- 'git -C ${HOME}/public/mdspan checkout 973ef6415a6396e5f0a55cb4c99afd1d1d541681'
	- 'git clone https://github.com/fwyzard/intro\_to\_alpaka.git -b tachep2025'
	- 'cd intro\_to\_alpaka/alpaka/'
	- 'make'
	- Write alpaka code in 'intro\_to\_alpaka/alpaka/' directory
	- Compile with 'nvcc -x cu --expt-relaxed-constexpr -std=c++20 -O2 -g -I${HOME}/public/alpaka/include -DALPAKA\_ACC\_GPU\_CUDA\_ENABLED alpakaMatrixOps.cu -o alpakaMatrixOps'
- Based on results of profiling the cuda code, I decided to use the standard 'matrix\_mult' function and the shared memory 'stencil\_2d' function
- Syntax of the kernels now starts with:
'''
struct stencil\_2d {
template <typename TAcc, typename T>
ALPAKA\_FN\_ACC void operator()(TAcc const& acc,  
				  T const\* __restrict__ in,  
				  T \* __restrict__ out,  
				 )const{  
	auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
	int threadIdxX = globalThreadIdx[0];
	int threadIdxY = globalThreadIdx[1];
	auto blocksize = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
	int blockdimX = blocksize[0];
	int blockdimY = blocksize[1];
	auto blockId = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
	int blockIdX = blockId[0];
	int blockIdY = blockId[1];	
}  
};
''' 


### Some things to remember :
- Include instructions on how you set-up the environment, compile and execute your C++/ CUDA/ Alpaka application.
- Save the output of the profiler for the results you will report in your project (in csv, txt or any other format you prefer).
