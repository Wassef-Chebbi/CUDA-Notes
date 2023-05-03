# TPCUDA

Useful ressources for discovering and working with Cuda



# Running Cuda programs on Colab
  To check the cuda installation, simply open a new notebook and set up the runtime configuration by going to ```Runtime``` >> ```Change runtime type``` and then setting the ```Hardware accelerator``` to  ```GPU```. 
  
 Secondaly type ```!nvcc --version```
  
  
  To enable CUDA programming and execution directly under Google Colab, you can install the nvcc4jupyter plugin as:
  
  ```!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git```
  
  
  
  After that, you should load the plugin as:
  
  ```%load_ext nvcc_plugin```
  
  
  To execute the code you should prefix it with: 
  ``` %%cu ```


# Writing Application Code for the GPU


Below is a .cu file (.cu is the file extension for CUDA-accelerated programs). It contains two functions, the first which will run on the CPU, the second which will run on the GPU. 
```
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```



```__global__ void GPUFunction()```

The __global__ keyword indicates that the following function will run on the GPU, and can be invoked globally, which in this context means either by the CPU, or, by the GPU.

Often, code executed on the CPU is referred to as **host** code, and code running on the GPU is referred to as **device** code.


```GPUFunction<<<1, 1>>>();```

Typically, when calling a function to run on the GPU, we call this function a **kernel**, which is launched.

<< >>configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (called blocks), as well as how many threads to execute in each block. notice the kernel is launching with 1 block of threads (the first execution configuration argument) which contains 1 thread (the second configuration argument).

```cudaDeviceSynchronize();```

Unlike much C/C++ code, launching kernels is asynchronous: the CPU code will continue to execute without waiting for the kernel launch to complete.
A call to ```cudaDeviceSynchronize```, a function provided by the CUDA runtime, will cause the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU.





# Launching Parallel Kernels


The execution configuration allows programmers to specify details about launching the kernel to run in parallel on multiple GPU threads. More precisely, the execution configuration allows programmers to specify how many groups of threads - called thread blocks, or just blocks - and how many threads they would like each thread block to contain. The syntax for this is:

```<<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>```

**The kernel code is executed by every thread in every thread block configured when the kernel is launched.**



Thus, under the assumption that a kernel called someKernel has been defined, the following are true:


```someKernel<<<1, 1>>>()``` 
is configured to run in a single thread block which has a single thread and will therefore run only once.


```someKernel<<<1, 10>>>()```
 is configured to run in a single thread block which has 10 threads and will therefore run 10 times.


```someKernel<<<10, 1>>>()```
 is configured to run in 10 thread blocks which each have a single thread and will therefore run 10 times.


```someKernel<<<10, 10>>>()```
 is configured to run in 10 thread blocks which each have 10 threads and will therefore run 100 times.





# Thread and Block Indices
Each thread is given an index within its thread block, starting at 0. Additionally, each block is given an index, starting at 0. 
Just as threads are grouped into thread blocks, blocks are grouped into a grid, which is the highest entity in the CUDA thread hierarchy. 
In summary, CUDA kernels are executed in a grid of 1 or more blocks, with each block containing the same number of 1 or more threads.

CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within. These variables are ```threadIdx.x``` and ```blockIdx.x``` respectively.





# Accelerating For Loops


For loops in CPU-only applications are ripe for acceleration: rather than run each iteration of the loop serially, each iteration of the loop can be run in parallel in its own thread. Consider the following for loop, and notice, though it is obvious, that it controls how many times the loop will execute, as well as defining what will happen for each iteration of the loop:

```
int N = 2<<20;
for (int i = 0; i < N; ++i)
{
  printf("%d\n", i);
}
```

In order to parallelize this loop, 2 steps must be taken:

1. A kernel must be written to do the work of a single iteration of the loop.


2. Because the kernel will be agnostic of other running kernels, the execution configuration must be such that the kernel executes the correct number of times, for example, the number of times the loop would have iterated.







# Using Block Dimensions for More Parallelization


There is a limit to the number of threads that can exist in a thread block: **1024** to be precise. In order to increase the amount of parallelism in accelerated applications, we must be able to coordinate among multiple thread blocks.

CUDA Kernels have access to a special variable that gives the number of threads in a block: **blockDim.x**. Using this variable, in conjunction with ```blockIdx.x``` and ```threadIdx.x```, increased parallelization can be accomplished by organizing parallel execution across multiple blocks of multiple threads with the idiomatic expression ```threadIdx.x + blockIdx.x * blockDim.x```. Here is a detailed example.



The execution configuration ```<<<10, 10>>>``` would launch a grid with a total of 100 threads, contained in 10 blocks of 10 threads. We would therefore hope for each thread to have the ability to calculate some index unique to itself between 0 and 99.


If block blockIdx.x equals 0, then blockIdx.x * blockDim.x is 0. Adding to 0 the possible threadIdx.x values 0 through 9, then we can generate the indices 0 through 9 within the 100 thread grid.


If block blockIdx.x equals 1, then blockIdx.x * blockDim.x is 10. Adding to 10 the possible threadIdx.x values 0 through 9, then we can generate the indices 10 through 19 within the 100 thread grid.


If block blockIdx.x equals 5, then blockIdx.x * blockDim.x is 50. Adding to 50 the possible threadIdx.x values 0 through 9, then we can generate the indices 50 through 59 within the 100 thread grid.


If block blockIdx.x equals 9, then blockIdx.x * blockDim.x is 90. Adding to 90 the possible threadIdx.x values 0 through 9, then we can generate the indices 90 through 99 within the 100 thread grid.





# Allocating Memory to be accessed on the GPU and the CPU



To allocate and free memory, and obtain a pointer that can be referenced in both host and device code, replace calls to ```malloc``` and ```free``` with ```cudaMallocManaged``` and cudaFree as in the following example:

```
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
```

```
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```


# Handling Block Configuration Mismatches to Number of Needed Threads

It may be the case that an execution configuration cannot be expressed that will create the exact number of threads needed for parallelizing a loop.

Blocks that contain a number of threads that are a multiple of 32 are often desirable for performance benefits. Assuming that we wanted to launch blocks each containing 256 threads (a multiple of 32), and needed to run 1000 parallel tasks, then there is no number of blocks that would produce an exact total of 1000 threads in the grid, since there is no integer value 32 can be multiplied by to equal exactly 1000.

This scenario can be easily addressed in the following way:

Write an execution configuration that creates more threads than necessary to perform the allotted work.
Pass a value as an argument into the kernel (N) that represents to the total size of the data set to be processed, or the total threads that are needed to complete the work.
After calculating the thread's index within the grid **(using tid+bid*bdim)**, **check that this index does not exceed N**, and only perform the pertinent work of the kernel if it does not.
Here is an examplee that ensures that there are always at least as many threads as needed for N, and only 1 additional block's worth of threads extra, at most:

```
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```

Because the execution configuration above results in more threads in the grid than N, care will need to be taken inside of the some_kernel definition so that some_kernel does not attempt to access out of range data elements, when being executed by one of the "extra" threads:

```
__global__ some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) // Check to make sure `idx` maps to some value within `N`
  {
    // Only do work if it does
  }
}
```


