
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

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
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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

unsigned int g_iterationNumber;

/* Agent Memory */

/* Person Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Person_list* d_Persons;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Person_list* d_Persons_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Person_list* d_Persons_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Person_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Person_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Person_values;  /**< Agent sort identifiers value */

/* Person state variables */
xmachine_memory_Person_list* h_Persons_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Person_list* d_Persons_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Person_default_count;   /**< Agent population size counter */ 

/* Person state variables */
xmachine_memory_Person_list* h_Persons_s2;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Person_list* d_Persons_s2;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Person_s2_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_Persons_default_variable_id_data_iteration;
unsigned int h_Persons_default_variable_age_data_iteration;
unsigned int h_Persons_default_variable_gender_data_iteration;
unsigned int h_Persons_default_variable_householdsize_data_iteration;
unsigned int h_Persons_s2_variable_id_data_iteration;
unsigned int h_Persons_s2_variable_age_data_iteration;
unsigned int h_Persons_s2_variable_gender_data_iteration;
unsigned int h_Persons_s2_variable_householdsize_data_iteration;


/* Message Memory */

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_Person;
size_t temp_scan_storage_bytes_Person;


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

/** Person_update
 * Agent function prototype for update function of Person agent
 */
void Person_update(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
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


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_Persons_default_variable_id_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    h_Persons_s2_variable_id_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_Person_SoA_size = sizeof(xmachine_memory_Person_list);
	h_Persons_default = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);
	h_Persons_s2 = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);

	/* Message memory allocation (CPU) */

	//Exit if agent or message buffer sizes are to small for function outputs
    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_Persons_default, &h_xmachine_memory_Person_default_count);
	

    PROFILE_PUSH_RANGE("allocate device");
	
	/* Person Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Persons, xmachine_Person_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Persons_swap, xmachine_Person_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Persons_new, xmachine_Person_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Person_keys, xmachine_memory_Person_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Person_values, xmachine_memory_Person_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Persons_default, xmachine_Person_SoA_size));
	gpuErrchk( cudaMemcpy( d_Persons_default, h_Persons_default, xmachine_Person_SoA_size, cudaMemcpyHostToDevice));
    
	/* s2 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Persons_s2, xmachine_Person_SoA_size));
	gpuErrchk( cudaMemcpy( d_Persons_s2, h_Persons_s2, xmachine_Person_SoA_size, cudaMemcpyHostToDevice));
    	
    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_Person = nullptr;
    temp_scan_storage_bytes_Person = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Person, 
        temp_scan_storage_bytes_Person, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Person_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Person, temp_scan_storage_bytes_Person));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
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

    PROFILE_POP_RANGE();

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
    initialiseHost();
    PROFILE_PUSH_RANGE("initialiseHost");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initialiseHost = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    generatePersonInit();
    PROFILE_PUSH_RANGE("generatePersonInit");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: generatePersonInit = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_Person_default_count: %u\n",get_agent_Person_default_count());
	
		printf("Init agent_Person_s2_count: %u\n",get_agent_Person_s2_count());
	
#endif
} 


void sort_Persons_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Person_default_count); 
	gridSize = (h_xmachine_memory_Person_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Person_keys, d_xmachine_memory_Person_values, d_Persons_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Person_keys),  thrust::device_pointer_cast(d_xmachine_memory_Person_keys) + h_xmachine_memory_Person_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Person_agents, no_sm, h_xmachine_memory_Person_default_count); 
	gridSize = (h_xmachine_memory_Person_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Person_values, d_Persons_default, d_Persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Person_list* d_Persons_temp = d_Persons_default;
	d_Persons_default = d_Persons_swap;
	d_Persons_swap = d_Persons_temp;	
}

void sort_Persons_s2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Person_s2_count); 
	gridSize = (h_xmachine_memory_Person_s2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Person_keys, d_xmachine_memory_Person_values, d_Persons_s2);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Person_keys),  thrust::device_pointer_cast(d_xmachine_memory_Person_keys) + h_xmachine_memory_Person_s2_count,  thrust::device_pointer_cast(d_xmachine_memory_Person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Person_agents, no_sm, h_xmachine_memory_Person_s2_count); 
	gridSize = (h_xmachine_memory_Person_s2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Person_values, d_Persons_s2, d_Persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Person_list* d_Persons_temp = d_Persons_s2;
	d_Persons_s2 = d_Persons_swap;
	d_Persons_swap = d_Persons_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

    exitFunction();
    PROFILE_PUSH_RANGE("exitFunction");
	PROFILE_POP_RANGE();

#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: exitFunction = %f (ms)\n", instrument_milliseconds);
#endif
	

	/* Agent data free*/
	
	/* Person Agent variables */
	gpuErrchk(cudaFree(d_Persons));
	gpuErrchk(cudaFree(d_Persons_swap));
	gpuErrchk(cudaFree(d_Persons_new));
	
	free( h_Persons_default);
	gpuErrchk(cudaFree(d_Persons_default));
	
	free( h_Persons_s2);
	gpuErrchk(cudaFree(d_Persons_s2));
	

	/* Message data free */
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Person));
    d_temp_scan_storage_Person = nullptr;
    temp_scan_storage_bytes_Person = 0;
    
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

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
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

	/* set all non partitioned and spatial partitioned message counts to 0*/

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_update");
	Person_update(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_update = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("generatePersonStep");
	generatePersonStep();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: generatePersonStep = %f (ms)\n", instrument_milliseconds);
#endif
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("customOutputStepFunction");
	customOutputStepFunction();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: customOutputStepFunction = %f (ms)\n", instrument_milliseconds);
#endif

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_Person_default_count: %u\n",get_agent_Person_default_count());
	
		printf("agent_Person_s2_count: %u\n",get_agent_Person_s2_count());
	
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
float h_env_TIME_STEP;
float h_env_SCALE_FACTOR;
unsigned int h_env_MAX_AGE;
unsigned int h_env_RANDOM_AGES;


//constant setter
void set_TIME_STEP(float* h_TIME_STEP){
    gpuErrchk(cudaMemcpyToSymbol(TIME_STEP, h_TIME_STEP, sizeof(float)));
    memcpy(&h_env_TIME_STEP, h_TIME_STEP,sizeof(float));
}

//constant getter
const float* get_TIME_STEP(){
    return &h_env_TIME_STEP;
}



//constant setter
void set_SCALE_FACTOR(float* h_SCALE_FACTOR){
    gpuErrchk(cudaMemcpyToSymbol(SCALE_FACTOR, h_SCALE_FACTOR, sizeof(float)));
    memcpy(&h_env_SCALE_FACTOR, h_SCALE_FACTOR,sizeof(float));
}

//constant getter
const float* get_SCALE_FACTOR(){
    return &h_env_SCALE_FACTOR;
}



//constant setter
void set_MAX_AGE(unsigned int* h_MAX_AGE){
    gpuErrchk(cudaMemcpyToSymbol(MAX_AGE, h_MAX_AGE, sizeof(unsigned int)));
    memcpy(&h_env_MAX_AGE, h_MAX_AGE,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_MAX_AGE(){
    return &h_env_MAX_AGE;
}



//constant setter
void set_RANDOM_AGES(unsigned int* h_RANDOM_AGES){
    gpuErrchk(cudaMemcpyToSymbol(RANDOM_AGES, h_RANDOM_AGES, sizeof(unsigned int)));
    memcpy(&h_env_RANDOM_AGES, h_RANDOM_AGES,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_RANDOM_AGES(){
    return &h_env_RANDOM_AGES;
}




/* Agent data access functions*/

    
int get_agent_Person_MAX_count(){
    return xmachine_memory_Person_MAX;
}


int get_agent_Person_default_count(){
	//continuous agent
	return h_xmachine_memory_Person_default_count;
	
}

xmachine_memory_Person_list* get_device_Person_default_agents(){
	return d_Persons_default;
}

xmachine_memory_Person_list* get_host_Person_default_agents(){
	return h_Persons_default;
}

int get_agent_Person_s2_count(){
	//continuous agent
	return h_xmachine_memory_Person_s2_count;
	
}

xmachine_memory_Person_list* get_device_Person_s2_agents(){
	return d_Persons_s2;
}

xmachine_memory_Person_list* get_host_Person_s2_agents(){
	return h_Persons_s2;
}



/* Host based access of agent variables*/

/** unsigned int get_Person_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_default_variable_id(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->id,
                    d_Persons_default->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_age(unsigned int index)
 * Gets the value of the age variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Person_default_variable_age(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_age_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->age,
                    d_Persons_default->age,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_age_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->age[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access age for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_gender(unsigned int index)
 * Gets the value of the gender variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable gender
 */
__host__ unsigned int get_Person_default_variable_gender(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_gender_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->gender,
                    d_Persons_default->gender,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_gender_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->gender[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access gender for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_householdsize(unsigned int index)
 * Gets the value of the householdsize variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdsize
 */
__host__ unsigned int get_Person_default_variable_householdsize(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_householdsize_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->householdsize,
                    d_Persons_default->householdsize,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_householdsize_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->householdsize[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access householdsize for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_s2_variable_id(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->id,
                    d_Persons_s2->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_age(unsigned int index)
 * Gets the value of the age variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Person_s2_variable_age(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_age_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->age,
                    d_Persons_s2->age,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_age_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->age[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access age for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_gender(unsigned int index)
 * Gets the value of the gender variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable gender
 */
__host__ unsigned int get_Person_s2_variable_gender(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_gender_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->gender,
                    d_Persons_s2->gender,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_gender_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->gender[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access gender for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_householdsize(unsigned int index)
 * Gets the value of the householdsize variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdsize
 */
__host__ unsigned int get_Person_s2_variable_householdsize(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_householdsize_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->householdsize,
                    d_Persons_s2->householdsize,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_householdsize_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->householdsize[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access householdsize for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_Person_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Person_hostToDevice(xmachine_memory_Person_list * d_dst, xmachine_memory_Person * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, &h_agent->age, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, &h_agent->gender, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, &h_agent->householdsize, sizeof(unsigned int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @todo - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_Person_hostToDevice(xmachine_memory_Person_list * d_dst, xmachine_memory_Person_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, h_src->age, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, h_src->gender, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, h_src->householdsize, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_Person* h_allocate_agent_Person(){
	xmachine_memory_Person* agent = (xmachine_memory_Person*)malloc(sizeof(xmachine_memory_Person));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Person));

    agent->age = 0;

	return agent;
}
void h_free_agent_Person(xmachine_memory_Person** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Person** h_allocate_agent_Person_array(unsigned int count){
	xmachine_memory_Person ** agents = (xmachine_memory_Person**)malloc(count * sizeof(xmachine_memory_Person*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Person();
	}
	return agents;
}
void h_free_agent_Person_array(xmachine_memory_Person*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Person(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Person_AoS_to_SoA(xmachine_memory_Person_list * dst, xmachine_memory_Person** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->age[i] = src[i]->age;
			 
			dst->gender[i] = src[i]->gender;
			 
			dst->householdsize[i] = src[i]->householdsize;
			
		}
	}
}


void h_add_agent_Person_default(xmachine_memory_Person* agent){
	if (h_xmachine_memory_Person_count + 1 > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of Person agents in state default will be exceeded by h_add_agent_Person_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Person_hostToDevice(d_Persons_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Person_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Person_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Persons_default, d_Persons_new, h_xmachine_memory_Person_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Person_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Persons_default_variable_id_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    

}
void h_add_agents_Person_default(xmachine_memory_Person** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Person_count + count > xmachine_memory_Person_MAX){
			printf("Error: Buffer size of Person agents in state default will be exceeded by h_add_agents_Person_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Person_AoS_to_SoA(h_Persons_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Person_hostToDevice(d_Persons_new, h_Persons_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Person_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Person_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Persons_default, d_Persons_new, h_xmachine_memory_Person_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Person_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Persons_default_variable_id_data_iteration = 0;
        h_Persons_default_variable_age_data_iteration = 0;
        h_Persons_default_variable_gender_data_iteration = 0;
        h_Persons_default_variable_householdsize_data_iteration = 0;
        

	}
}


void h_add_agent_Person_s2(xmachine_memory_Person* agent){
	if (h_xmachine_memory_Person_count + 1 > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of Person agents in state s2 will be exceeded by h_add_agent_Person_s2\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Person_hostToDevice(d_Persons_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Person_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Person_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Persons_s2, d_Persons_new, h_xmachine_memory_Person_s2_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Person_s2_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Persons_s2_variable_id_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    

}
void h_add_agents_Person_s2(xmachine_memory_Person** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Person_count + count > xmachine_memory_Person_MAX){
			printf("Error: Buffer size of Person agents in state s2 will be exceeded by h_add_agents_Person_s2\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Person_AoS_to_SoA(h_Persons_s2, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Person_hostToDevice(d_Persons_new, h_Persons_s2, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Person_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Person_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Persons_s2, d_Persons_new, h_xmachine_memory_Person_s2_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Person_s2_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Persons_s2_variable_id_data_iteration = 0;
        h_Persons_s2_variable_age_data_iteration = 0;
        h_Persons_s2_variable_gender_data_iteration = 0;
        h_Persons_s2_variable_householdsize_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

unsigned int reduce_Person_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->id),  thrust::device_pointer_cast(d_Persons_default->id) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->id),  thrust::device_pointer_cast(d_Persons_default->id) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_age_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->age),  thrust::device_pointer_cast(d_Persons_default->age) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_age_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->age),  thrust::device_pointer_cast(d_Persons_default->age) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_age_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->age);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_age_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->age);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_gender_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->gender),  thrust::device_pointer_cast(d_Persons_default->gender) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_gender_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->gender),  thrust::device_pointer_cast(d_Persons_default->gender) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_gender_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->gender);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_gender_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->gender);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_householdsize_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->householdsize),  thrust::device_pointer_cast(d_Persons_default->householdsize) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_householdsize_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->householdsize),  thrust::device_pointer_cast(d_Persons_default->householdsize) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_householdsize_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->householdsize);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_householdsize_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->householdsize);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->id),  thrust::device_pointer_cast(d_Persons_s2->id) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->id),  thrust::device_pointer_cast(d_Persons_s2->id) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_age_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->age),  thrust::device_pointer_cast(d_Persons_s2->age) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_age_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->age),  thrust::device_pointer_cast(d_Persons_s2->age) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_age_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->age);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_age_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->age);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_gender_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->gender),  thrust::device_pointer_cast(d_Persons_s2->gender) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_gender_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->gender),  thrust::device_pointer_cast(d_Persons_s2->gender) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_gender_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->gender);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_gender_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->gender);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_householdsize_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->householdsize),  thrust::device_pointer_cast(d_Persons_s2->householdsize) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_householdsize_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->householdsize),  thrust::device_pointer_cast(d_Persons_s2->householdsize) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_householdsize_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->householdsize);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_householdsize_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->householdsize);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int Person_update_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Person_update
 * Agent function prototype for update function of Person agent
 */
void Person_update(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Person_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Person_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Person_list* Persons_default_temp = d_Persons;
	d_Persons = d_Persons_default;
	d_Persons_default = Persons_default_temp;
	//set working count to current state count
	h_xmachine_memory_Person_count = h_xmachine_memory_Person_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_count, &h_xmachine_memory_Person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Person_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_update, Person_update_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_update_sm_size(blockSize);
	
	
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Person_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Person_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Persons);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (update)
	//Reallocate   : true
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_update<<<g, b, sm_size, stream>>>(d_Persons, d_rand48);
	gpuErrchkLaunch();
	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Person, 
        temp_scan_storage_bytes_Person, 
        d_Persons->_scan_input,
        d_Persons->_position,
        h_xmachine_memory_Person_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_swap, d_Persons, 0, h_xmachine_memory_Person_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_Person_list* update_Persons_temp = d_Persons;
	d_Persons = d_Persons_swap;
	d_Persons_swap = update_Persons_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Persons_swap->_position[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Persons_swap->_scan_input[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Person_count = scan_last_sum+1;
	else
		h_xmachine_memory_Person_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_count, &h_xmachine_memory_Person_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_default_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of update agents in state default will be exceeded moving working agents to next state in function update\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Person_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_default, d_Persons, h_xmachine_memory_Person_default_count, h_xmachine_memory_Person_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Person_default_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));	
	
	
}


 
extern void reset_Person_default_count()
{
    h_xmachine_memory_Person_default_count = 0;
}
 
extern void reset_Person_s2_count()
{
    h_xmachine_memory_Person_s2_count = 0;
}
