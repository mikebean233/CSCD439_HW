#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>

#include "timing.c"

#define UPB RAND_MAX 
#define LEVELS 5
#define MAXDRET 102400
#define THRESH 2 

float * fillArray(int n, int upbound)
{
   int i;
   
   float *ret = (float *)malloc(sizeof(float) * n );

   /* Intializes random number generator */
   //seeds the random number generator used by the function rand.
   srand(time(NULL));

   /* generate n random numbers from 0 to unbound - 1 */
   for( i = 0 ; i < n ; i++ ) {
      ret[i] = rand() % upbound * 1.0f;
   }
   
   return ret;
}

void printArray(float *arr, int n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%5.0f ", arr[i]);

   printf("\n");
}

float cpuReduce(float *h_in, int n)
{
    float m = - FLT_MAX;
    int i;
    for(i = 0; i < n; i ++)
    {
        if(h_in[i] > m)
            m = h_in[i];
    }
    return m;
}


__global__ void reduce2(float *in, float *out, int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            if(sdata[tid] < sdata[tid + s])
            	sdata[tid] = sdata[tid + s]; //bigger number stored in low index
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce3(float *in, float *out, int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0;
    int activeCount = n / 2;
    int leftData  = 0;
    int rightData = 0;

    for(; activeCount > 0; activeCount /= 2)
    {
        leftData  = sdata[tid];
        rightData = sdata[tid + activeCount];
        if(tid < activeCount && rightData > leftData)
            sdata[tid] = rightData;
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void usage()
{
   printf("Usage: ./progName blockWidth numElementsInput p \n");
}


int main(int argc, char *argv[])
{
   // create a large workload so we can easily measure the
   // performance difference on CPU and GPU

   // to run this program: ./a.out blockWidth numElements p
   int shouldPrint = 0;
   if(argc < 3 || argc > 4) {
      usage();
      return 1;
   } else  if(argc == 3){
         shouldPrint = 0;
   } else if(argv[3][0]=='p'){
         shouldPrint=1;
   } else {
         usage();
         return 1;
   }
  
   //
   int tile_width = atoi(argv[1]);
   if ( ! tile_width )
   {
       printf("Wrong argument passed in for blockWidth!\n");
       exit(-1);
   }
   int n = atoi(argv[2]); //size of 1D input array
   if ( ! n )
   {
       printf("Wrong argument passed in for size of input array!\n");
       exit(-1);
   }

   // set up host memory
   float *h_in, *h_out, *d_in, *d_out;
   //int sizeDout[LEVELS]; //we can have at most 5 levels of kernel launch
   h_out = (float *)malloc(MAXDRET * sizeof(float));
   memset(h_out, 0, MAXDRET * sizeof(float));

   //generate input data from random generator
   h_in = fillArray(n, UPB);

   if( ! h_in || ! h_out )
   {
       printf("Error in host memory allocation!\n");
       exit(-1);
   }
   int num_block = ceil(n / (float)tile_width);
   dim3 block(tile_width, 1, 1);
   dim3 grid(num_block, 1, 1);

   // allocate storage for the device
   cudaMalloc((void**)&d_in, sizeof(float) * n);
   cudaMalloc((void**)&d_out, sizeof(float) * MAXDRET);
   cudaMemset(d_out, 0, sizeof(float) * MAXDRET);

   // copy input to the device
   cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

   // time the kernel launches using CUDA events
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);
 
   printf("The input array is:\n"); 
   //print out original array
   if(shouldPrint)
       printArray(h_in, n); 


   //----------------------time many kernel launches and take the average time--------------------
   const size_t num_launches = 10;
   float average_simple_time = 0;
   int num_in = n, num_out = ceil((float)n / tile_width);
   int launch = 1;

   printf("Timing simple GPU implementation… \n");
   for(int i = 0; i < num_launches; ++i)
   {
       // record a CUDA event immediately before and after the kernel launch
       cudaEventRecord(launch_begin,0);
       //reduce2<<<grid, block, tile_width * sizeof(float)>>>(d_in, d_out, n);
       //cudaMemcpy(h_out, d_out, sizeof(float) * num_block, cudaMemcpyDeviceToHost);
       while( 1 )
       {
           if(launch % 2 == 1) // odd launch
               reduce2<<<grid, block, tile_width * sizeof(float)>>>(d_in, d_out, num_in);
           else
               reduce3<<<grid, block, tile_width * sizeof(float)>>>(d_out, d_in, num_in);

           cudaDeviceSynchronize();
           
           // if the number of local max returned by kernel is greater than the threshold,
           // we do reduction on GPU for these returned local maxes for another pass,
           // until, num_out < threshold
           if(num_out >= THRESH) 
           {
               num_in = num_out;
               num_out = ceil((float)num_out / tile_width);
               grid.x = num_out; //change the grid dimension in x direction
               //cudaMemset(d_in, 0, n * sizeof(float));//reset d_in, used for output of next iteration 
           }
           else
           {
               //copy the ouput of last launch back to host,
               if(launch % 2 == 1)
                  cudaMemcpy(h_out, d_out, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
               else
                  cudaMemcpy(h_out, d_in, sizeof(float) * num_out, cudaMemcpyDeviceToHost);
               
               break;
           }
           launch ++;
       }//end of while
    
       cudaEventRecord(launch_end,0);
       cudaEventSynchronize(launch_end);

       // measure the time spent in the kernel
       float time = 0;
       cudaEventElapsedTime(&time, launch_begin, launch_end);

       average_simple_time += time;
  }
  average_simple_time /= num_launches;
  printf(" done! GPU time cost in second: %f\n", average_simple_time / 1000);

  printf("The output array from device is:\n");
  //if(shouldPrint)
      printArray(h_out, num_out); 


  //------------------------ now time the sequential code on CPU------------------------------

  // time many multiplication calls and take the average time
  float average_cpu_time = 0;
  clock_t now, then;
  int num_cpu_test = 3;
  float max = 0;

  printf("Timing CPU implementation…\n");
  for(int i = 0; i < num_cpu_test; ++i) //launch 3 times on CPU
  {
    // timing on CPU
    then = clock();
    max = cpuReduce(h_in, n);
    now = clock();
   
    // measure the time spent on CPU
    float time = 0;
    time = timeCost(then, now);

    average_cpu_time += time;
  }
  average_cpu_time /= num_cpu_test;
  printf(" done. CPU time cost in second: %f\n", average_cpu_time);
  
  //if (shouldPrint)
      printf("CPU finding max number is %.1f\n", max);

  //--------------------------------clean up-----------------------------------------------------
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);
  
  free(h_in);
  free(h_out);

  return 0;
}

