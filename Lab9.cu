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

#define SIZE 3 //Original 3.
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void seriedeWallis(double n)
{
	double i;
	double pi = 4;

	//printf("El número de iteraciones es: %1.16f\n", n);

	for(i = 3; i<= (n + 2); i+=2){
		pi = pi * ((i - 1) / i) * ((i + 1) / i);
		printf("Valor aproximado del PI: %1.16f\n", pi);
	}
	//printf(%1.16f, *pi);
}

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void seriedeNilakantha(double n)
{
	double n, i;
	double pi = 3; 
	int s = 1;  
	 for(i = 2; i <= n*2; i += 2){
		 pi = pi + s * (4 / (i * (i + 1) * (i + 2)));
 		 s = -s;
 	}

}

int main(void) 
{
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);

	double n, *i;         // Number of iterations and control variable 
   	double *pi;

	printf("Aproximando el valor de pi por medio de la serie de Wallis");
	printf("\nIngrese el número de iteraciones: ");
	scanf("%lf", &n);
	printf("\nPor favor espere.....");
	
	//seriedeWallis<<<1, SIZE, 0, stream1>>>(n,i,pi);
	
	int *a1, *b1, *c1; // host vars to use in stream 1 mem ptrs
	
	double *dev_a1, *dev_b1, *dev_c1; // stream 1 mem ptrs
	
	//stream 1 - mem allocation at Global memmory for device and host, in order
	cudaMalloc( (void**)&dev_a1, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_b1, SIZE * sizeof(int) );
	cudaMalloc( (void**)&dev_c1, SIZE * sizeof(int) );
	
	cudaHostAlloc((void**)&a1,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&b1,SIZE*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&c1,SIZE*sizeof(int),cudaHostAllocDefault);
	
	// generate data
	for(int i=0;i<SIZE;i++) 
	{
	a1[i] = 1+i;
	b1[i] = 5+i;

	}
	
	for(int i=0;i < SIZE;i++)
	{ // loop over data in chunks
	// stream 1
	//cudaMemcpyAsync(n,pi,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream1);
	//cudaMemcpyAsync(dev_b1,b1,SIZE*sizeof(int),cudaMemcpyHostToDevice,stream1);
	seriedeWallis<<<1,SIZE,0,stream1>>>(n); //Mandando al kernel la info.
	printf("Si llegué");
	//cudaMemcpyAsync(c1,dev_c1,SIZE*sizeof(int),cudaMemcpyDeviceToHost,stream1);

	}
	cudaStreamSynchronize(stream1); // wait for stream1 to finish
	
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

	return 0;
}