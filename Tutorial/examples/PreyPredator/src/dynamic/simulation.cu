
/*
 * FLAME GPU v 1.4.0 for CUDA 7.5
 * Copyright 2015 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

//Disable internal thrust warnings about conversions
#pragma warning(push)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#pragma warning(pop)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* prey Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_prey_list* d_preys;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_prey_list* d_preys_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_prey_list* d_preys_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_prey_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_prey_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_prey_values;  /**< Agent sort identifiers value */
    
/* prey state variables */
xmachine_memory_prey_list* h_preys_default1;      /**< Pointer to agent list (population) on host*/
xmachine_memory_prey_list* d_preys_default1;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_prey_default1_count;   /**< Agent population size counter */ 

/* predator Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_predator_list* d_predators;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_predator_list* d_predators_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_predator_list* d_predators_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_predator_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_predator_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_predator_values;  /**< Agent sort identifiers value */
    
/* predator state variables */
xmachine_memory_predator_list* h_predators_default2;      /**< Pointer to agent list (population) on host*/
xmachine_memory_predator_list* d_predators_default2;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_predator_default2_count;   /**< Agent population size counter */ 

/* grass Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_grass_list* d_grasss;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_grass_list* d_grasss_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_grass_list* d_grasss_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_grass_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_grass_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_grass_values;  /**< Agent sort identifiers value */
    
/* grass state variables */
xmachine_memory_grass_list* h_grasss_default3;      /**< Pointer to agent list (population) on host*/
xmachine_memory_grass_list* d_grasss_default3;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_grass_default3_count;   /**< Agent population size counter */ 


/* Message Memory */

/* grass_location Message variables */
xmachine_message_grass_location_list* h_grass_locations;         /**< Pointer to message list on host*/
xmachine_message_grass_location_list* d_grass_locations;         /**< Pointer to message list on device*/
xmachine_message_grass_location_list* d_grass_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_grass_location_count;         /**< message list counter*/
int h_message_grass_location_output_type;   /**< message output type (single or optional)*/

/* prey_location Message variables */
xmachine_message_prey_location_list* h_prey_locations;         /**< Pointer to message list on host*/
xmachine_message_prey_location_list* d_prey_locations;         /**< Pointer to message list on device*/
xmachine_message_prey_location_list* d_prey_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_prey_location_count;         /**< message list counter*/
int h_message_prey_location_output_type;   /**< message output type (single or optional)*/

/* pred_location Message variables */
xmachine_message_pred_location_list* h_pred_locations;         /**< Pointer to message list on host*/
xmachine_message_pred_location_list* d_pred_locations;         /**< Pointer to message list on device*/
xmachine_message_pred_location_list* d_pred_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_pred_location_count;         /**< message list counter*/
int h_message_pred_location_output_type;   /**< message output type (single or optional)*/

/* prey_eaten Message variables */
xmachine_message_prey_eaten_list* h_prey_eatens;         /**< Pointer to message list on host*/
xmachine_message_prey_eaten_list* d_prey_eatens;         /**< Pointer to message list on device*/
xmachine_message_prey_eaten_list* d_prey_eatens_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_prey_eaten_count;         /**< message list counter*/
int h_message_prey_eaten_output_type;   /**< message output type (single or optional)*/

/* grass_eaten Message variables */
xmachine_message_grass_eaten_list* h_grass_eatens;         /**< Pointer to message list on host*/
xmachine_message_grass_eaten_list* d_grass_eatens;         /**< Pointer to message list on device*/
xmachine_message_grass_eaten_list* d_grass_eatens_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_grass_eaten_count;         /**< message list counter*/
int h_message_grass_eaten_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** prey_prey_output_location
 * Agent function prototype for prey_output_location function of prey agent
 */
void prey_prey_output_location(cudaStream_t &stream);

/** prey_prey_avoid_pred
 * Agent function prototype for prey_avoid_pred function of prey agent
 */
void prey_prey_avoid_pred(cudaStream_t &stream);

/** prey_prey_flock
 * Agent function prototype for prey_flock function of prey agent
 */
void prey_prey_flock(cudaStream_t &stream);

/** prey_prey_move
 * Agent function prototype for prey_move function of prey agent
 */
void prey_prey_move(cudaStream_t &stream);

/** prey_prey_eaten
 * Agent function prototype for prey_eaten function of prey agent
 */
void prey_prey_eaten(cudaStream_t &stream);

/** prey_prey_eat_or_starve
 * Agent function prototype for prey_eat_or_starve function of prey agent
 */
void prey_prey_eat_or_starve(cudaStream_t &stream);

/** prey_prey_reproduction
 * Agent function prototype for prey_reproduction function of prey agent
 */
void prey_prey_reproduction(cudaStream_t &stream);

/** predator_pred_output_location
 * Agent function prototype for pred_output_location function of predator agent
 */
void predator_pred_output_location(cudaStream_t &stream);

/** predator_pred_follow_prey
 * Agent function prototype for pred_follow_prey function of predator agent
 */
void predator_pred_follow_prey(cudaStream_t &stream);

/** predator_pred_avoid
 * Agent function prototype for pred_avoid function of predator agent
 */
void predator_pred_avoid(cudaStream_t &stream);

/** predator_pred_move
 * Agent function prototype for pred_move function of predator agent
 */
void predator_pred_move(cudaStream_t &stream);

/** predator_pred_eat_or_starve
 * Agent function prototype for pred_eat_or_starve function of predator agent
 */
void predator_pred_eat_or_starve(cudaStream_t &stream);

/** predator_pred_reproduction
 * Agent function prototype for pred_reproduction function of predator agent
 */
void predator_pred_reproduction(cudaStream_t &stream);

/** grass_grass_output_location
 * Agent function prototype for grass_output_location function of grass agent
 */
void grass_grass_output_location(cudaStream_t &stream);

/** grass_grass_eaten
 * Agent function prototype for grass_eaten function of grass agent
 */
void grass_grass_eaten(cudaStream_t &stream);

/** grass_grass_growth
 * Agent function prototype for grass_growth function of grass agent
 */
void grass_grass_growth(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  

	printf("Allocating Host and Device memory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_prey_SoA_size = sizeof(xmachine_memory_prey_list);
	h_preys_default1 = (xmachine_memory_prey_list*)malloc(xmachine_prey_SoA_size);
	int xmachine_predator_SoA_size = sizeof(xmachine_memory_predator_list);
	h_predators_default2 = (xmachine_memory_predator_list*)malloc(xmachine_predator_SoA_size);
	int xmachine_grass_SoA_size = sizeof(xmachine_memory_grass_list);
	h_grasss_default3 = (xmachine_memory_grass_list*)malloc(xmachine_grass_SoA_size);

	/* Message memory allocation (CPU) */
	int message_grass_location_SoA_size = sizeof(xmachine_message_grass_location_list);
	h_grass_locations = (xmachine_message_grass_location_list*)malloc(message_grass_location_SoA_size);
	int message_prey_location_SoA_size = sizeof(xmachine_message_prey_location_list);
	h_prey_locations = (xmachine_message_prey_location_list*)malloc(message_prey_location_SoA_size);
	int message_pred_location_SoA_size = sizeof(xmachine_message_pred_location_list);
	h_pred_locations = (xmachine_message_pred_location_list*)malloc(message_pred_location_SoA_size);
	int message_prey_eaten_SoA_size = sizeof(xmachine_message_prey_eaten_list);
	h_prey_eatens = (xmachine_message_prey_eaten_list*)malloc(message_prey_eaten_SoA_size);
	int message_grass_eaten_SoA_size = sizeof(xmachine_message_grass_eaten_list);
	h_grass_eatens = (xmachine_message_grass_eaten_list*)malloc(message_grass_eaten_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

	//read initial states
	readInitialStates(inputfile, h_preys_default1, &h_xmachine_memory_prey_default1_count, h_predators_default2, &h_xmachine_memory_predator_default2_count, h_grasss_default3, &h_xmachine_memory_grass_default3_count);
	
	
	/* prey Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_preys, xmachine_prey_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_preys_swap, xmachine_prey_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_preys_new, xmachine_prey_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_prey_keys, xmachine_memory_prey_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_prey_values, xmachine_memory_prey_MAX* sizeof(uint)));
	/* default1 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_preys_default1, xmachine_prey_SoA_size));
	gpuErrchk( cudaMemcpy( d_preys_default1, h_preys_default1, xmachine_prey_SoA_size, cudaMemcpyHostToDevice));
    
	/* predator Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_predators, xmachine_predator_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_predators_swap, xmachine_predator_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_predators_new, xmachine_predator_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_predator_keys, xmachine_memory_predator_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_predator_values, xmachine_memory_predator_MAX* sizeof(uint)));
	/* default2 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_predators_default2, xmachine_predator_SoA_size));
	gpuErrchk( cudaMemcpy( d_predators_default2, h_predators_default2, xmachine_predator_SoA_size, cudaMemcpyHostToDevice));
    
	/* grass Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_grasss, xmachine_grass_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_grasss_swap, xmachine_grass_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_grasss_new, xmachine_grass_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_grass_keys, xmachine_memory_grass_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_grass_values, xmachine_memory_grass_MAX* sizeof(uint)));
	/* default3 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_grasss_default3, xmachine_grass_SoA_size));
	gpuErrchk( cudaMemcpy( d_grasss_default3, h_grasss_default3, xmachine_grass_SoA_size, cudaMemcpyHostToDevice));
    
	/* grass_location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_grass_locations, message_grass_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_grass_locations_swap, message_grass_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_grass_locations, h_grass_locations, message_grass_location_SoA_size, cudaMemcpyHostToDevice));
	
	/* prey_location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_prey_locations, message_prey_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_prey_locations_swap, message_prey_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_prey_locations, h_prey_locations, message_prey_location_SoA_size, cudaMemcpyHostToDevice));
	
	/* pred_location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_pred_locations, message_pred_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_pred_locations_swap, message_pred_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_pred_locations, h_pred_locations, message_pred_location_SoA_size, cudaMemcpyHostToDevice));
	
	/* prey_eaten Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_prey_eatens, message_prey_eaten_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_prey_eatens_swap, message_prey_eaten_SoA_size));
	gpuErrchk( cudaMemcpy( d_prey_eatens, h_prey_eatens, message_prey_eaten_SoA_size, cudaMemcpyHostToDevice));
	
	/* grass_eaten Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_grass_eatens, message_grass_eaten_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_grass_eatens_swap, message_grass_eaten_SoA_size));
	gpuErrchk( cudaMemcpy( d_grass_eatens, h_grass_eatens, message_grass_eaten_SoA_size, cudaMemcpyHostToDevice));
		

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	initLogFile();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initLogFile = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
  gpuErrchk(cudaStreamCreate(&stream2));
  gpuErrchk(cudaStreamCreate(&stream3));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_prey_default1_count: %u\n",get_agent_prey_default1_count());
	
		printf("Init agent_predator_default2_count: %u\n",get_agent_predator_default2_count());
	
		printf("Init agent_grass_default3_count: %u\n",get_agent_grass_default3_count());
	
#endif
} 


void sort_preys_default1(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_prey_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_prey_default1_count); 
	gridSize = (h_xmachine_memory_prey_default1_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_prey_keys, d_xmachine_memory_prey_values, d_preys_default1);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_prey_keys),  thrust::device_pointer_cast(d_xmachine_memory_prey_keys) + h_xmachine_memory_prey_default1_count,  thrust::device_pointer_cast(d_xmachine_memory_prey_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_prey_agents, no_sm, h_xmachine_memory_prey_default1_count); 
	gridSize = (h_xmachine_memory_prey_default1_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_prey_agents<<<gridSize, blockSize>>>(d_xmachine_memory_prey_values, d_preys_default1, d_preys_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_prey_list* d_preys_temp = d_preys_default1;
	d_preys_default1 = d_preys_swap;
	d_preys_swap = d_preys_temp;	
}

void sort_predators_default2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_predator_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_predator_default2_count); 
	gridSize = (h_xmachine_memory_predator_default2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_predator_keys, d_xmachine_memory_predator_values, d_predators_default2);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_predator_keys),  thrust::device_pointer_cast(d_xmachine_memory_predator_keys) + h_xmachine_memory_predator_default2_count,  thrust::device_pointer_cast(d_xmachine_memory_predator_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_predator_agents, no_sm, h_xmachine_memory_predator_default2_count); 
	gridSize = (h_xmachine_memory_predator_default2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_predator_agents<<<gridSize, blockSize>>>(d_xmachine_memory_predator_values, d_predators_default2, d_predators_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_predator_list* d_predators_temp = d_predators_default2;
	d_predators_default2 = d_predators_swap;
	d_predators_swap = d_predators_temp;	
}

void sort_grasss_default3(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_grass_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_grass_default3_count); 
	gridSize = (h_xmachine_memory_grass_default3_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_grass_keys, d_xmachine_memory_grass_values, d_grasss_default3);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_grass_keys),  thrust::device_pointer_cast(d_xmachine_memory_grass_keys) + h_xmachine_memory_grass_default3_count,  thrust::device_pointer_cast(d_xmachine_memory_grass_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_grass_agents, no_sm, h_xmachine_memory_grass_default3_count); 
	gridSize = (h_xmachine_memory_grass_default3_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_grass_agents<<<gridSize, blockSize>>>(d_xmachine_memory_grass_values, d_grasss_default3, d_grasss_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_grass_list* d_grasss_temp = d_grasss_default3;
	d_grasss_default3 = d_grasss_swap;
	d_grasss_swap = d_grasss_temp;	
}


void cleanup(){

    /* Call all exit functions */
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

	closeLogFile();
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: closeLogFile = %f (ms)\n", instrument_milliseconds);
#endif
	

	/* Agent data free*/
	
	/* prey Agent variables */
	gpuErrchk(cudaFree(d_preys));
	gpuErrchk(cudaFree(d_preys_swap));
	gpuErrchk(cudaFree(d_preys_new));
	
	free( h_preys_default1);
	gpuErrchk(cudaFree(d_preys_default1));
	
	/* predator Agent variables */
	gpuErrchk(cudaFree(d_predators));
	gpuErrchk(cudaFree(d_predators_swap));
	gpuErrchk(cudaFree(d_predators_new));
	
	free( h_predators_default2);
	gpuErrchk(cudaFree(d_predators_default2));
	
	/* grass Agent variables */
	gpuErrchk(cudaFree(d_grasss));
	gpuErrchk(cudaFree(d_grasss_swap));
	gpuErrchk(cudaFree(d_grasss_new));
	
	free( h_grasss_default3);
	gpuErrchk(cudaFree(d_grasss_default3));
	

	/* Message data free */
	
	/* grass_location Message variables */
	free( h_grass_locations);
	gpuErrchk(cudaFree(d_grass_locations));
	gpuErrchk(cudaFree(d_grass_locations_swap));
	
	/* prey_location Message variables */
	free( h_prey_locations);
	gpuErrchk(cudaFree(d_prey_locations));
	gpuErrchk(cudaFree(d_prey_locations_swap));
	
	/* pred_location Message variables */
	free( h_pred_locations);
	gpuErrchk(cudaFree(d_pred_locations));
	gpuErrchk(cudaFree(d_pred_locations_swap));
	
	/* prey_eaten Message variables */
	free( h_prey_eatens);
	gpuErrchk(cudaFree(d_prey_eatens));
	gpuErrchk(cudaFree(d_prey_eatens_swap));
	
	/* grass_eaten Message variables */
	free( h_grass_eatens);
	gpuErrchk(cudaFree(d_grass_eatens));
	gpuErrchk(cudaFree(d_grass_eatens_swap));
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

	/* set all non partitioned and spatial partitioned message counts to 0*/
	h_message_grass_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_grass_location_count, &h_message_grass_location_count, sizeof(int)));
	
	h_message_prey_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_prey_location_count, &h_message_prey_location_count, sizeof(int)));
	
	h_message_pred_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_pred_location_count, &h_message_pred_location_count, sizeof(int)));
	
	h_message_prey_eaten_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_prey_eaten_count, &h_message_prey_eaten_count, sizeof(int)));
	
	h_message_grass_eaten_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_grass_eaten_count, &h_message_grass_eaten_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_output_location(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_output_location = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_output_location(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_output_location = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	grass_grass_output_location(stream3);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: grass_grass_output_location = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_follow_prey(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_follow_prey = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_avoid_pred(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_avoid_pred = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_flock(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_flock = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_avoid(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_avoid = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_move(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_move = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_move(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_move = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	grass_grass_eaten(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: grass_grass_eaten = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_eaten(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_eaten = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_eat_or_starve(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_eat_or_starve = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_eat_or_starve(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_eat_or_starve = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	predator_pred_reproduction(stream1);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: predator_pred_reproduction = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	prey_prey_reproduction(stream2);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: prey_prey_reproduction = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	grass_grass_growth(stream3);
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: grass_grass_growth = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	outputToLogFile();
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: outputToLogFile = %f (ms)\n", instrument_milliseconds);
#endif

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_prey_default1_count: %u\n",get_agent_prey_default1_count());
	
		printf("agent_predator_default2_count: %u\n",get_agent_predator_default2_count());
	
		printf("agent_grass_default3_count: %u\n",get_agent_grass_default3_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration
float h_env_REPRODUCE_PREY_PROB;
float h_env_REPRODUCE_PREDATOR_PROB;
int h_env_GAIN_FROM_FOOD_PREDATOR;
int h_env_GAIN_FROM_FOOD_PREY;
int h_env_GRASS_REGROW_CYCLES;


//constant setter
void set_REPRODUCE_PREY_PROB(float* h_REPRODUCE_PREY_PROB){
    gpuErrchk(cudaMemcpyToSymbol(REPRODUCE_PREY_PROB, h_REPRODUCE_PREY_PROB, sizeof(float)));
    memcpy(&h_env_REPRODUCE_PREY_PROB, h_REPRODUCE_PREY_PROB,sizeof(float));
}


//constant getter
const float* get_REPRODUCE_PREY_PROB(){
    return &h_env_REPRODUCE_PREY_PROB;
}


//constant setter
void set_REPRODUCE_PREDATOR_PROB(float* h_REPRODUCE_PREDATOR_PROB){
    gpuErrchk(cudaMemcpyToSymbol(REPRODUCE_PREDATOR_PROB, h_REPRODUCE_PREDATOR_PROB, sizeof(float)));
    memcpy(&h_env_REPRODUCE_PREDATOR_PROB, h_REPRODUCE_PREDATOR_PROB,sizeof(float));
}


//constant getter
const float* get_REPRODUCE_PREDATOR_PROB(){
    return &h_env_REPRODUCE_PREDATOR_PROB;
}


//constant setter
void set_GAIN_FROM_FOOD_PREDATOR(int* h_GAIN_FROM_FOOD_PREDATOR){
    gpuErrchk(cudaMemcpyToSymbol(GAIN_FROM_FOOD_PREDATOR, h_GAIN_FROM_FOOD_PREDATOR, sizeof(int)));
    memcpy(&h_env_GAIN_FROM_FOOD_PREDATOR, h_GAIN_FROM_FOOD_PREDATOR,sizeof(int));
}


//constant getter
const int* get_GAIN_FROM_FOOD_PREDATOR(){
    return &h_env_GAIN_FROM_FOOD_PREDATOR;
}


//constant setter
void set_GAIN_FROM_FOOD_PREY(int* h_GAIN_FROM_FOOD_PREY){
    gpuErrchk(cudaMemcpyToSymbol(GAIN_FROM_FOOD_PREY, h_GAIN_FROM_FOOD_PREY, sizeof(int)));
    memcpy(&h_env_GAIN_FROM_FOOD_PREY, h_GAIN_FROM_FOOD_PREY,sizeof(int));
}


//constant getter
const int* get_GAIN_FROM_FOOD_PREY(){
    return &h_env_GAIN_FROM_FOOD_PREY;
}


//constant setter
void set_GRASS_REGROW_CYCLES(int* h_GRASS_REGROW_CYCLES){
    gpuErrchk(cudaMemcpyToSymbol(GRASS_REGROW_CYCLES, h_GRASS_REGROW_CYCLES, sizeof(int)));
    memcpy(&h_env_GRASS_REGROW_CYCLES, h_GRASS_REGROW_CYCLES,sizeof(int));
}


//constant getter
const int* get_GRASS_REGROW_CYCLES(){
    return &h_env_GRASS_REGROW_CYCLES;
}



/* Agent data access functions*/

    
int get_agent_prey_MAX_count(){
    return xmachine_memory_prey_MAX;
}


int get_agent_prey_default1_count(){
	//continuous agent
	return h_xmachine_memory_prey_default1_count;
	
}

xmachine_memory_prey_list* get_device_prey_default1_agents(){
	return d_preys_default1;
}

xmachine_memory_prey_list* get_host_prey_default1_agents(){
	return h_preys_default1;
}

    
int get_agent_predator_MAX_count(){
    return xmachine_memory_predator_MAX;
}


int get_agent_predator_default2_count(){
	//continuous agent
	return h_xmachine_memory_predator_default2_count;
	
}

xmachine_memory_predator_list* get_device_predator_default2_agents(){
	return d_predators_default2;
}

xmachine_memory_predator_list* get_host_predator_default2_agents(){
	return h_predators_default2;
}

    
int get_agent_grass_MAX_count(){
    return xmachine_memory_grass_MAX;
}


int get_agent_grass_default3_count(){
	//continuous agent
	return h_xmachine_memory_grass_default3_count;
	
}

xmachine_memory_grass_list* get_device_grass_default3_agents(){
	return d_grasss_default3;
}

xmachine_memory_grass_list* get_host_grass_default3_agents(){
	return h_grasss_default3;
}



/*  Analytics Functions */

int reduce_prey_default1_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->id),  thrust::device_pointer_cast(d_preys_default1->id) + h_xmachine_memory_prey_default1_count);
}
int count_prey_default1_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_preys_default1->id),  thrust::device_pointer_cast(d_preys_default1->id) + h_xmachine_memory_prey_default1_count, count_value);
}
float reduce_prey_default1_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->x),  thrust::device_pointer_cast(d_preys_default1->x) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->y),  thrust::device_pointer_cast(d_preys_default1->y) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_type_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->type),  thrust::device_pointer_cast(d_preys_default1->type) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->fx),  thrust::device_pointer_cast(d_preys_default1->fx) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->fy),  thrust::device_pointer_cast(d_preys_default1->fy) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_steer_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->steer_x),  thrust::device_pointer_cast(d_preys_default1->steer_x) + h_xmachine_memory_prey_default1_count);
}
float reduce_prey_default1_steer_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->steer_y),  thrust::device_pointer_cast(d_preys_default1->steer_y) + h_xmachine_memory_prey_default1_count);
}
int reduce_prey_default1_life_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_preys_default1->life),  thrust::device_pointer_cast(d_preys_default1->life) + h_xmachine_memory_prey_default1_count);
}
int count_prey_default1_life_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_preys_default1->life),  thrust::device_pointer_cast(d_preys_default1->life) + h_xmachine_memory_prey_default1_count, count_value);
}
int reduce_predator_default2_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->id),  thrust::device_pointer_cast(d_predators_default2->id) + h_xmachine_memory_predator_default2_count);
}
int count_predator_default2_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_predators_default2->id),  thrust::device_pointer_cast(d_predators_default2->id) + h_xmachine_memory_predator_default2_count, count_value);
}
float reduce_predator_default2_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->x),  thrust::device_pointer_cast(d_predators_default2->x) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->y),  thrust::device_pointer_cast(d_predators_default2->y) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_type_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->type),  thrust::device_pointer_cast(d_predators_default2->type) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_fx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->fx),  thrust::device_pointer_cast(d_predators_default2->fx) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_fy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->fy),  thrust::device_pointer_cast(d_predators_default2->fy) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_steer_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->steer_x),  thrust::device_pointer_cast(d_predators_default2->steer_x) + h_xmachine_memory_predator_default2_count);
}
float reduce_predator_default2_steer_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->steer_y),  thrust::device_pointer_cast(d_predators_default2->steer_y) + h_xmachine_memory_predator_default2_count);
}
int reduce_predator_default2_life_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_predators_default2->life),  thrust::device_pointer_cast(d_predators_default2->life) + h_xmachine_memory_predator_default2_count);
}
int count_predator_default2_life_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_predators_default2->life),  thrust::device_pointer_cast(d_predators_default2->life) + h_xmachine_memory_predator_default2_count, count_value);
}
int reduce_grass_default3_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->id),  thrust::device_pointer_cast(d_grasss_default3->id) + h_xmachine_memory_grass_default3_count);
}
int count_grass_default3_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_grasss_default3->id),  thrust::device_pointer_cast(d_grasss_default3->id) + h_xmachine_memory_grass_default3_count, count_value);
}
float reduce_grass_default3_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->x),  thrust::device_pointer_cast(d_grasss_default3->x) + h_xmachine_memory_grass_default3_count);
}
float reduce_grass_default3_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->y),  thrust::device_pointer_cast(d_grasss_default3->y) + h_xmachine_memory_grass_default3_count);
}
float reduce_grass_default3_type_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->type),  thrust::device_pointer_cast(d_grasss_default3->type) + h_xmachine_memory_grass_default3_count);
}
int reduce_grass_default3_dead_cycles_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->dead_cycles),  thrust::device_pointer_cast(d_grasss_default3->dead_cycles) + h_xmachine_memory_grass_default3_count);
}
int count_grass_default3_dead_cycles_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_grasss_default3->dead_cycles),  thrust::device_pointer_cast(d_grasss_default3->dead_cycles) + h_xmachine_memory_grass_default3_count, count_value);
}
int reduce_grass_default3_available_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_grasss_default3->available),  thrust::device_pointer_cast(d_grasss_default3->available) + h_xmachine_memory_grass_default3_count);
}
int count_grass_default3_available_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_grasss_default3->available),  thrust::device_pointer_cast(d_grasss_default3->available) + h_xmachine_memory_grass_default3_count, count_value);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int prey_prey_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** prey_prey_output_location
 * Agent function prototype for prey_output_location function of prey agent
 */
void prey_prey_output_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_prey_location_count + h_xmachine_memory_prey_count > xmachine_message_prey_location_MAX){
		printf("Error: Buffer size of prey_location message will be exceeded in function prey_output_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_output_location, prey_prey_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_prey_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_prey_location_output_type, &h_message_prey_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : prey_location
	//Agent Output : 
	GPUFLAME_prey_output_location<<<g, b, sm_size, stream>>>(d_preys, d_prey_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_prey_location_count += h_xmachine_memory_prey_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_prey_location_count, &h_message_prey_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_output_location agents in state default1 will be exceeded moving working agents to next state in function prey_output_location\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_avoid_pred_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_pred_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** prey_prey_avoid_pred
 * Agent function prototype for prey_avoid_pred function of prey agent
 */
void prey_prey_avoid_pred(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_avoid_pred, prey_prey_avoid_pred_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_avoid_pred_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_avoid_pred)
	//Reallocate   : false
	//Input        : pred_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_prey_avoid_pred<<<g, b, sm_size, stream>>>(d_preys, d_pred_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_avoid_pred agents in state default1 will be exceeded moving working agents to next state in function prey_avoid_pred\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_flock_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_prey_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** prey_prey_flock
 * Agent function prototype for prey_flock function of prey agent
 */
void prey_prey_flock(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_flock, prey_prey_flock_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_flock_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_flock)
	//Reallocate   : false
	//Input        : prey_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_prey_flock<<<g, b, sm_size, stream>>>(d_preys, d_prey_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_flock agents in state default1 will be exceeded moving working agents to next state in function prey_flock\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** prey_prey_move
 * Agent function prototype for prey_move function of prey agent
 */
void prey_prey_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_move, prey_prey_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_move_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_move)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_prey_move<<<g, b, sm_size, stream>>>(d_preys);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_move agents in state default1 will be exceeded moving working agents to next state in function prey_move\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_eaten_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_pred_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** prey_prey_eaten
 * Agent function prototype for prey_eaten function of prey agent
 */
void prey_prey_eaten(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_prey_eaten_count + h_xmachine_memory_prey_count > xmachine_message_prey_eaten_MAX){
		printf("Error: Buffer size of prey_eaten message will be exceeded in function prey_eaten\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_eaten, prey_prey_eaten_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_eaten_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_prey_eaten_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_prey_eaten_output_type, &h_message_prey_eaten_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_prey_eaten_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_prey_eaten_swaps<<<gridSize, blockSize, 0, stream>>>(d_prey_eatens); 
	gpuErrchkLaunch();
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_prey_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_prey_scan_input<<<gridSize, blockSize, 0, stream>>>(d_preys);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_eaten)
	//Reallocate   : true
	//Input        : pred_location
	//Output       : prey_eaten
	//Agent Output : 
	GPUFLAME_prey_eaten<<<g, b, sm_size, stream>>>(d_preys, d_pred_locations, d_prey_eatens);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//prey_eaten Message Type Prefix Sum
	
	//swap output
	xmachine_message_prey_eaten_list* d_prey_eatens_scanswap_temp = d_prey_eatens;
	d_prey_eatens = d_prey_eatens_swap;
	d_prey_eatens_swap = d_prey_eatens_scanswap_temp;
	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_prey_eatens_swap->_scan_input), thrust::device_pointer_cast(d_prey_eatens_swap->_scan_input) + h_xmachine_memory_prey_count, thrust::device_pointer_cast(d_prey_eatens_swap->_position));
	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_prey_eaten_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_prey_eaten_messages<<<gridSize, blockSize, 0, stream>>>(d_prey_eatens, d_prey_eatens_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_prey_eatens_swap->_position[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_prey_eatens_swap->_scan_input[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_prey_eaten_count += scan_last_sum+1;
	}else{
		h_message_prey_eaten_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_prey_eaten_count, &h_message_prey_eaten_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_preys->_scan_input), thrust::device_pointer_cast(d_preys->_scan_input) + h_xmachine_memory_prey_count, thrust::device_pointer_cast(d_preys->_position));
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_swap, d_preys, 0, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_prey_list* prey_eaten_preys_temp = d_preys;
	d_preys = d_preys_swap;
	d_preys_swap = prey_eaten_preys_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_preys_swap->_position[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_preys_swap->_scan_input[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_prey_count = scan_last_sum+1;
	else
		h_xmachine_memory_prey_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_eaten agents in state default1 will be exceeded moving working agents to next state in function prey_eaten\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_eat_or_starve_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_grass_eaten));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** prey_prey_eat_or_starve
 * Agent function prototype for prey_eat_or_starve function of prey agent
 */
void prey_prey_eat_or_starve(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_eat_or_starve, prey_prey_eat_or_starve_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_eat_or_starve_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_prey_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_prey_scan_input<<<gridSize, blockSize, 0, stream>>>(d_preys);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_eat_or_starve)
	//Reallocate   : true
	//Input        : grass_eaten
	//Output       : 
	//Agent Output : 
	GPUFLAME_prey_eat_or_starve<<<g, b, sm_size, stream>>>(d_preys, d_grass_eatens);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_preys->_scan_input), thrust::device_pointer_cast(d_preys->_scan_input) + h_xmachine_memory_prey_count, thrust::device_pointer_cast(d_preys->_position));
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_swap, d_preys, 0, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_prey_list* prey_eat_or_starve_preys_temp = d_preys;
	d_preys = d_preys_swap;
	d_preys_swap = prey_eat_or_starve_preys_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_preys_swap->_position[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_preys_swap->_scan_input[h_xmachine_memory_prey_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_prey_count = scan_last_sum+1;
	else
		h_xmachine_memory_prey_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_eat_or_starve agents in state default1 will be exceeded moving working agents to next state in function prey_eat_or_starve\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int prey_prey_reproduction_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** prey_prey_reproduction
 * Agent function prototype for prey_reproduction function of prey agent
 */
void prey_prey_reproduction(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_prey_default1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_prey_default1_count;

	
	//FOR prey AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_prey_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_prey_scan_input<<<gridSize, blockSize, 0, stream>>>(d_preys_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_prey_list* preys_default1_temp = d_preys;
	d_preys = d_preys_default1;
	d_preys_default1 = preys_default1_temp;
	//set working count to current state count
	h_xmachine_memory_prey_count = h_xmachine_memory_prey_default1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_count, &h_xmachine_memory_prey_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_prey_default1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prey_reproduction, prey_prey_reproduction_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = prey_prey_reproduction_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (prey_reproduction)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : prey
	GPUFLAME_prey_reproduction<<<g, b, sm_size, stream>>>(d_preys, d_preys_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE prey AGENTS ARE KILLED (needed for scatter)
	int preys_pre_death_count = h_xmachine_memory_prey_count;
	
	//FOR prey AGENT OUTPUT SCATTER AGENTS 
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_preys_new->_scan_input), thrust::device_pointer_cast(d_preys_new->_scan_input) + preys_pre_death_count, thrust::device_pointer_cast(d_preys_new->_position));
	//reset agent count
	int prey_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_preys_new->_position[preys_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_preys_new->_scan_input[preys_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		prey_after_birth_count = h_xmachine_memory_prey_default1_count + scan_last_sum+1;
	else
		prey_after_birth_count = h_xmachine_memory_prey_default1_count + scan_last_sum;
	//check buffer is not exceeded
	if (prey_after_birth_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey agents in state default1 will be exceeded writing new agents in function prey_reproduction\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys_new, h_xmachine_memory_prey_default1_count, preys_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_prey_default1_count = prey_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_prey_default1_count+h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
		printf("Error: Buffer size of prey_reproduction agents in state default1 will be exceeded moving working agents to next state in function prey_reproduction\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_prey_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_prey_Agents<<<gridSize, blockSize, 0, stream>>>(d_preys_default1, d_preys, h_xmachine_memory_prey_default1_count, h_xmachine_memory_prey_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_prey_default1_count += h_xmachine_memory_prey_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_prey_default1_count, &h_xmachine_memory_prey_default1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** predator_pred_output_location
 * Agent function prototype for pred_output_location function of predator agent
 */
void predator_pred_output_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_pred_location_count + h_xmachine_memory_predator_count > xmachine_message_pred_location_MAX){
		printf("Error: Buffer size of pred_location message will be exceeded in function pred_output_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_output_location, predator_pred_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_pred_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_pred_location_output_type, &h_message_pred_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : pred_location
	//Agent Output : 
	GPUFLAME_pred_output_location<<<g, b, sm_size, stream>>>(d_predators, d_pred_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_pred_location_count += h_xmachine_memory_predator_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_pred_location_count, &h_message_pred_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_output_location agents in state default2 will be exceeded moving working agents to next state in function pred_output_location\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_follow_prey_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_prey_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** predator_pred_follow_prey
 * Agent function prototype for pred_follow_prey function of predator agent
 */
void predator_pred_follow_prey(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_follow_prey, predator_pred_follow_prey_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_follow_prey_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_follow_prey)
	//Reallocate   : false
	//Input        : prey_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_pred_follow_prey<<<g, b, sm_size, stream>>>(d_predators, d_prey_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_follow_prey agents in state default2 will be exceeded moving working agents to next state in function pred_follow_prey\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_avoid_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_pred_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** predator_pred_avoid
 * Agent function prototype for pred_avoid function of predator agent
 */
void predator_pred_avoid(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_avoid, predator_pred_avoid_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_avoid_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_avoid)
	//Reallocate   : false
	//Input        : pred_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_pred_avoid<<<g, b, sm_size, stream>>>(d_predators, d_pred_locations);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_avoid agents in state default2 will be exceeded moving working agents to next state in function pred_avoid\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** predator_pred_move
 * Agent function prototype for pred_move function of predator agent
 */
void predator_pred_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_move, predator_pred_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_move_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_move)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_pred_move<<<g, b, sm_size, stream>>>(d_predators);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_move agents in state default2 will be exceeded moving working agents to next state in function pred_move\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_eat_or_starve_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_prey_eaten));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** predator_pred_eat_or_starve
 * Agent function prototype for pred_eat_or_starve function of predator agent
 */
void predator_pred_eat_or_starve(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_eat_or_starve, predator_pred_eat_or_starve_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_eat_or_starve_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_predator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_predator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_predators);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_eat_or_starve)
	//Reallocate   : true
	//Input        : prey_eaten
	//Output       : 
	//Agent Output : 
	GPUFLAME_pred_eat_or_starve<<<g, b, sm_size, stream>>>(d_predators, d_prey_eatens);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_predators->_scan_input), thrust::device_pointer_cast(d_predators->_scan_input) + h_xmachine_memory_predator_count, thrust::device_pointer_cast(d_predators->_position));
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_swap, d_predators, 0, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_predator_list* pred_eat_or_starve_predators_temp = d_predators;
	d_predators = d_predators_swap;
	d_predators_swap = pred_eat_or_starve_predators_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_predators_swap->_position[h_xmachine_memory_predator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_predators_swap->_scan_input[h_xmachine_memory_predator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_predator_count = scan_last_sum+1;
	else
		h_xmachine_memory_predator_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_eat_or_starve agents in state default2 will be exceeded moving working agents to next state in function pred_eat_or_starve\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int predator_pred_reproduction_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** predator_pred_reproduction
 * Agent function prototype for pred_reproduction function of predator agent
 */
void predator_pred_reproduction(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_predator_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_predator_default2_count;

	
	//FOR predator AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_predator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_predator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_predators_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_predator_list* predators_default2_temp = d_predators;
	d_predators = d_predators_default2;
	d_predators_default2 = predators_default2_temp;
	//set working count to current state count
	h_xmachine_memory_predator_count = h_xmachine_memory_predator_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_count, &h_xmachine_memory_predator_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_predator_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_pred_reproduction, predator_pred_reproduction_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = predator_pred_reproduction_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (pred_reproduction)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : predator
	GPUFLAME_pred_reproduction<<<g, b, sm_size, stream>>>(d_predators, d_predators_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE predator AGENTS ARE KILLED (needed for scatter)
	int predators_pre_death_count = h_xmachine_memory_predator_count;
	
	//FOR predator AGENT OUTPUT SCATTER AGENTS 
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_predators_new->_scan_input), thrust::device_pointer_cast(d_predators_new->_scan_input) + predators_pre_death_count, thrust::device_pointer_cast(d_predators_new->_position));
	//reset agent count
	int predator_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_predators_new->_position[predators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_predators_new->_scan_input[predators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		predator_after_birth_count = h_xmachine_memory_predator_default2_count + scan_last_sum+1;
	else
		predator_after_birth_count = h_xmachine_memory_predator_default2_count + scan_last_sum;
	//check buffer is not exceeded
	if (predator_after_birth_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of predator agents in state default2 will be exceeded writing new agents in function pred_reproduction\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators_new, h_xmachine_memory_predator_default2_count, predators_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_predator_default2_count = predator_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_predator_default2_count+h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
		printf("Error: Buffer size of pred_reproduction agents in state default2 will be exceeded moving working agents to next state in function pred_reproduction\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_predator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_predator_Agents<<<gridSize, blockSize, 0, stream>>>(d_predators_default2, d_predators, h_xmachine_memory_predator_default2_count, h_xmachine_memory_predator_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_predator_default2_count += h_xmachine_memory_predator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_predator_default2_count, &h_xmachine_memory_predator_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int grass_grass_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** grass_grass_output_location
 * Agent function prototype for grass_output_location function of grass agent
 */
void grass_grass_output_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_grass_default3_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_grass_default3_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_grass_list* grasss_default3_temp = d_grasss;
	d_grasss = d_grasss_default3;
	d_grasss_default3 = grasss_default3_temp;
	//set working count to current state count
	h_xmachine_memory_grass_count = h_xmachine_memory_grass_default3_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_count, &h_xmachine_memory_grass_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_grass_default3_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_grass_location_count + h_xmachine_memory_grass_count > xmachine_message_grass_location_MAX){
		printf("Error: Buffer size of grass_location message will be exceeded in function grass_output_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_grass_output_location, grass_grass_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = grass_grass_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_grass_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_grass_location_output_type, &h_message_grass_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (grass_output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : grass_location
	//Agent Output : 
	GPUFLAME_grass_output_location<<<g, b, sm_size, stream>>>(d_grasss, d_grass_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_grass_location_count += h_xmachine_memory_grass_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_grass_location_count, &h_message_grass_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_grass_default3_count+h_xmachine_memory_grass_count > xmachine_memory_grass_MAX){
		printf("Error: Buffer size of grass_output_location agents in state default3 will be exceeded moving working agents to next state in function grass_output_location\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_grass_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_grass_Agents<<<gridSize, blockSize, 0, stream>>>(d_grasss_default3, d_grasss, h_xmachine_memory_grass_default3_count, h_xmachine_memory_grass_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_grass_default3_count += h_xmachine_memory_grass_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int grass_grass_eaten_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_prey_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** grass_grass_eaten
 * Agent function prototype for grass_eaten function of grass agent
 */
void grass_grass_eaten(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_grass_default3_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_grass_default3_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_grass_list* grasss_default3_temp = d_grasss;
	d_grasss = d_grasss_default3;
	d_grasss_default3 = grasss_default3_temp;
	//set working count to current state count
	h_xmachine_memory_grass_count = h_xmachine_memory_grass_default3_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_count, &h_xmachine_memory_grass_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_grass_default3_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_grass_eaten_count + h_xmachine_memory_grass_count > xmachine_message_grass_eaten_MAX){
		printf("Error: Buffer size of grass_eaten message will be exceeded in function grass_eaten\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_grass_eaten, grass_grass_eaten_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = grass_grass_eaten_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_grass_eaten_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_grass_eaten_output_type, &h_message_grass_eaten_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_grass_eaten_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_grass_eaten_swaps<<<gridSize, blockSize, 0, stream>>>(d_grass_eatens); 
	gpuErrchkLaunch();
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_grass_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_grass_scan_input<<<gridSize, blockSize, 0, stream>>>(d_grasss);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (grass_eaten)
	//Reallocate   : true
	//Input        : prey_location
	//Output       : grass_eaten
	//Agent Output : 
	GPUFLAME_grass_eaten<<<g, b, sm_size, stream>>>(d_grasss, d_prey_locations, d_grass_eatens);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//grass_eaten Message Type Prefix Sum
	
	//swap output
	xmachine_message_grass_eaten_list* d_grass_eatens_scanswap_temp = d_grass_eatens;
	d_grass_eatens = d_grass_eatens_swap;
	d_grass_eatens_swap = d_grass_eatens_scanswap_temp;
	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_grass_eatens_swap->_scan_input), thrust::device_pointer_cast(d_grass_eatens_swap->_scan_input) + h_xmachine_memory_grass_count, thrust::device_pointer_cast(d_grass_eatens_swap->_position));
	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_grass_eaten_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_grass_eaten_messages<<<gridSize, blockSize, 0, stream>>>(d_grass_eatens, d_grass_eatens_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_grass_eatens_swap->_position[h_xmachine_memory_grass_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_grass_eatens_swap->_scan_input[h_xmachine_memory_grass_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_grass_eaten_count += scan_last_sum+1;
	}else{
		h_message_grass_eaten_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_grass_eaten_count, &h_message_grass_eaten_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_grasss->_scan_input), thrust::device_pointer_cast(d_grasss->_scan_input) + h_xmachine_memory_grass_count, thrust::device_pointer_cast(d_grasss->_position));
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_grass_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_grass_Agents<<<gridSize, blockSize, 0, stream>>>(d_grasss_swap, d_grasss, 0, h_xmachine_memory_grass_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_grass_list* grass_eaten_grasss_temp = d_grasss;
	d_grasss = d_grasss_swap;
	d_grasss_swap = grass_eaten_grasss_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_grasss_swap->_position[h_xmachine_memory_grass_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_grasss_swap->_scan_input[h_xmachine_memory_grass_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_grass_count = scan_last_sum+1;
	else
		h_xmachine_memory_grass_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_count, &h_xmachine_memory_grass_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_grass_default3_count+h_xmachine_memory_grass_count > xmachine_memory_grass_MAX){
		printf("Error: Buffer size of grass_eaten agents in state default3 will be exceeded moving working agents to next state in function grass_eaten\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_grass_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_grass_Agents<<<gridSize, blockSize, 0, stream>>>(d_grasss_default3, d_grasss, h_xmachine_memory_grass_default3_count, h_xmachine_memory_grass_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_grass_default3_count += h_xmachine_memory_grass_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int grass_grass_growth_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** grass_grass_growth
 * Agent function prototype for grass_growth function of grass agent
 */
void grass_grass_growth(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_grass_default3_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_grass_default3_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_grass_list* grasss_default3_temp = d_grasss;
	d_grasss = d_grasss_default3;
	d_grasss_default3 = grasss_default3_temp;
	//set working count to current state count
	h_xmachine_memory_grass_count = h_xmachine_memory_grass_default3_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_count, &h_xmachine_memory_grass_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_grass_default3_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_grass_growth, grass_grass_growth_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = grass_grass_growth_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (grass_growth)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_grass_growth<<<g, b, sm_size, stream>>>(d_grasss, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_grass_default3_count+h_xmachine_memory_grass_count > xmachine_memory_grass_MAX){
		printf("Error: Buffer size of grass_growth agents in state default3 will be exceeded moving working agents to next state in function grass_growth\n");
		exit(EXIT_FAILURE);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_grass_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_grass_Agents<<<gridSize, blockSize, 0, stream>>>(d_grasss_default3, d_grasss, h_xmachine_memory_grass_default3_count, h_xmachine_memory_grass_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_grass_default3_count += h_xmachine_memory_grass_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_grass_default3_count, &h_xmachine_memory_grass_default3_count, sizeof(int)));	
	
	
}


 
extern void reset_prey_default1_count()
{
    h_xmachine_memory_prey_default1_count = 0;
}
 
extern void reset_predator_default2_count()
{
    h_xmachine_memory_predator_default2_count = 0;
}
 
extern void reset_grass_default3_count()
{
    h_xmachine_memory_grass_default3_count = 0;
}
