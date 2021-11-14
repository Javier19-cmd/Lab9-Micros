/**
 * --------------------------------------------------------
 * Universidad del Valle de Guatemala
 * CC3056 - Programación de Microprocesadores
 * --------------------------------------------------------
 * addVectors_grid.cu
 * --------------------------------------------------------
 * Suma de dos vectores en CUDA
 * Demuestra la forma de usar CUDA 7 Streams para 
 * concurrencia simplificada
 * --------------------------------------------------------
 * AUTH.	Mark Harris
 * MODIF.   Kimberly B.
 * DATE		2021-11-7
 * --------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SIZE 3
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void seriedeWallis(double *i, double *n, double *pi)
{

	for(*i = 3.0; *i<= (*n + 2.0); *i+=2.0){
		*pi = 4.0;
		*pi = *pi * ((*i - 1.0) / *i) * ((*i + 1.0) / *i);
		printf("Valor aproximado del PI: %1.16f\n", *pi);
	}
}

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void Kernel2( int *d, int *e, int *f)
{
	int myID = threadIdx.x + blockDim.x * blockIdx.x;
	// Solo trabajan N hilos
	if (myID < SIZE)
	{
		f[myID] = d[myID] * e[myID];
	}
}

int main(void) 
{
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	double *n, *i;         // Number of iterations and control variable 
   	double *pi;

	printf("Aproximando el valor de pi por medio de la serie de Wallis");
	printf("\nIngrese el número de iteraciones: ");
	scanf("%lf", &n);
	printf("\nPor favor espere.....");

	for(*i = 3.0; *i<=(*n + 2.0); *i+=2.0){
		*pi = 4.0;

		*pi = *pi * ((*i - 1.0) / *i) * ((*i + 1.0) / *i);

		printf("%d",*pi);
	}
	
	//seriedeWallis<<<1, SIZE, 0, stream1>>>(n,i,pi);
	
	int *a1, *b1, *c1; // host vars to use in stream 1 mem ptrs
	int *a2, *b2, *c2; // host vars to use in stream 2 mem ptrs
	
	int *dev_a1, *dev_b1, *dev_c1; // stream 1 mem ptrs
	int *dev_a2, *dev_b2, *dev_c2; // stream 2 mem ptrs
	
	//stream 1 - mem allocation at Global memmory for device and host, in order
	cudaMalloc( (void**)&dev_a1, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_b1, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_c1, SIZE * sizeof(int) );
	
	cudaHostAlloc((void**)&a1,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&b1,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&c1,SIZE*sizeof(int),cudaHostAllocDefault);
	
	//stream 2 - mem allocation at Global memmory for device and host, in order
	cudaMalloc( (void**)&dev_a2, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_b2, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_c2, SIZE * sizeof(int) );
	
	cudaHostAlloc((void**)&a2,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&b2,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&c2,SIZE*sizeof(int),cudaHostAllocDefault);

	// generate data
	for(int i=0;i<SIZE;i++) 
	{
	a1[i] = 1+i;
	b1[i] = 5+i;
	
	a2[i] = 3+i;
	b2[i] = 4+i;
	}
	
	for(int i=0;i < SIZE;i++)
	{ // loop over data in chunks
	// stream 1
	cudaMemcpyAsync(dev_a1,a1,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream1);
	cudaMemcpyAsync(dev_b1,b1,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream1);
	//Kernel1<<<1,SIZE,0,stream1>>>(dev_a1,dev_b1,dev_c1);
	cudaMemcpyAsync(c1,dev_c1,SIZE*sizeof(int),cudaMemcpyDeviceToHost,stream1);
	

	//stream 2
	cudaMemcpyAsync(dev_a2,a2,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream2);
	cudaMemcpyAsync(dev_b2,b2,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream2);
	//Kernel2<<<1,SIZE,1,stream2>>>(dev_a2,dev_b2,dev_c2);
	cudaMemcpyAsync(c2,dev_c2,SIZE*sizeof(int),cudaMemcpyDeviceToHost,stream2);
	}
	cudaStreamSynchronize(stream1); // wait for stream1 to finish
	cudaStreamSynchronize(stream2); // wait for stream2 to finish
	
	/*
	printf("--- STREAM 1 ---\n");
	printf( "> Vector a1:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", a1[i]);
	}

	printf( "> \n Vector b1:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", b1[i]);
	}

	printf( "> \n Vector c1:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", c1[i]);
	}
	printf( "--- STREAM 2 ---\n");

	printf("\n");
	printf( "> Vector a2:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", a2[i]);
	}

	printf( "> \n Vector b2:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", b2[i]);
	}

	printf( "> \n Vector c2:\n");
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d ", c2[i]);
	}
	printf( "\n");
*/
	cudaStreamDestroy(stream1); 		// because we care
	cudaStreamDestroy(stream2); 

	return 0;
}