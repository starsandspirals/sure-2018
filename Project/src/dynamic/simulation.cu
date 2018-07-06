
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

/* Household Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Household_list* d_Households;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Household_list* d_Households_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Household_list* d_Households_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Household_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Household_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Household_values;  /**< Agent sort identifiers value */

/* Household state variables */
xmachine_memory_Household_list* h_Households_hhdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Household_list* d_Households_hhdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Household_hhdefault_count;   /**< Agent population size counter */ 

/* HouseholdMembership Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_HouseholdMembership_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_HouseholdMembership_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_HouseholdMembership_values;  /**< Agent sort identifiers value */

/* HouseholdMembership state variables */
xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships_hhmembershipdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_hhmembershipdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;   /**< Agent population size counter */ 

/* Church Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Church_list* d_Churchs;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Church_list* d_Churchs_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Church_list* d_Churchs_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Church_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Church_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Church_values;  /**< Agent sort identifiers value */

/* Church state variables */
xmachine_memory_Church_list* h_Churchs_chudefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Church_list* d_Churchs_chudefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Church_chudefault_count;   /**< Agent population size counter */ 

/* ChurchMembership Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_ChurchMembership_list* d_ChurchMemberships;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_ChurchMembership_list* d_ChurchMemberships_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_ChurchMembership_list* d_ChurchMemberships_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_ChurchMembership_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_ChurchMembership_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_ChurchMembership_values;  /**< Agent sort identifiers value */

/* ChurchMembership state variables */
xmachine_memory_ChurchMembership_list* h_ChurchMemberships_chumembershipdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_ChurchMembership_list* d_ChurchMemberships_chumembershipdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_ChurchMembership_chumembershipdefault_count;   /**< Agent population size counter */ 

/* Transport Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Transport_list* d_Transports;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Transport_list* d_Transports_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Transport_list* d_Transports_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Transport_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Transport_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Transport_values;  /**< Agent sort identifiers value */

/* Transport state variables */
xmachine_memory_Transport_list* h_Transports_trdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Transport_list* d_Transports_trdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Transport_trdefault_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_Persons_default_variable_id_data_iteration;
unsigned int h_Persons_default_variable_step_data_iteration;
unsigned int h_Persons_default_variable_age_data_iteration;
unsigned int h_Persons_default_variable_gender_data_iteration;
unsigned int h_Persons_default_variable_householdsize_data_iteration;
unsigned int h_Persons_default_variable_churchfreq_data_iteration;
unsigned int h_Persons_default_variable_churchdur_data_iteration;
unsigned int h_Persons_default_variable_transportuser_data_iteration;
unsigned int h_Persons_default_variable_transportfreq_data_iteration;
unsigned int h_Persons_default_variable_transportdur_data_iteration;
unsigned int h_Persons_default_variable_household_data_iteration;
unsigned int h_Persons_default_variable_church_data_iteration;
unsigned int h_Persons_default_variable_busy_data_iteration;
unsigned int h_Persons_default_variable_startstep_data_iteration;
unsigned int h_Persons_s2_variable_id_data_iteration;
unsigned int h_Persons_s2_variable_step_data_iteration;
unsigned int h_Persons_s2_variable_age_data_iteration;
unsigned int h_Persons_s2_variable_gender_data_iteration;
unsigned int h_Persons_s2_variable_householdsize_data_iteration;
unsigned int h_Persons_s2_variable_churchfreq_data_iteration;
unsigned int h_Persons_s2_variable_churchdur_data_iteration;
unsigned int h_Persons_s2_variable_transportuser_data_iteration;
unsigned int h_Persons_s2_variable_transportfreq_data_iteration;
unsigned int h_Persons_s2_variable_transportdur_data_iteration;
unsigned int h_Persons_s2_variable_household_data_iteration;
unsigned int h_Persons_s2_variable_church_data_iteration;
unsigned int h_Persons_s2_variable_busy_data_iteration;
unsigned int h_Persons_s2_variable_startstep_data_iteration;
unsigned int h_Households_hhdefault_variable_id_data_iteration;
unsigned int h_Households_hhdefault_variable_step_data_iteration;
unsigned int h_Households_hhdefault_variable_size_data_iteration;
unsigned int h_Households_hhdefault_variable_people_data_iteration;
unsigned int h_Households_hhdefault_variable_churchgoing_data_iteration;
unsigned int h_Households_hhdefault_variable_churchfreq_data_iteration;
unsigned int h_Households_hhdefault_variable_adults_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration;
unsigned int h_Churchs_chudefault_variable_id_data_iteration;
unsigned int h_Churchs_chudefault_variable_step_data_iteration;
unsigned int h_Churchs_chudefault_variable_size_data_iteration;
unsigned int h_Churchs_chudefault_variable_duration_data_iteration;
unsigned int h_Churchs_chudefault_variable_households_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration;
unsigned int h_Transports_trdefault_variable_id_data_iteration;
unsigned int h_Transports_trdefault_variable_step_data_iteration;
unsigned int h_Transports_trdefault_variable_duration_data_iteration;


/* Message Memory */

/* household_membership Message variables */
xmachine_message_household_membership_list* h_household_memberships;         /**< Pointer to message list on host*/
xmachine_message_household_membership_list* d_household_memberships;         /**< Pointer to message list on device*/
xmachine_message_household_membership_list* d_household_memberships_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_household_membership_count;         /**< message list counter*/
int h_message_household_membership_output_type;   /**< message output type (single or optional)*/

/* church_membership Message variables */
xmachine_message_church_membership_list* h_church_memberships;         /**< Pointer to message list on host*/
xmachine_message_church_membership_list* d_church_memberships;         /**< Pointer to message list on device*/
xmachine_message_church_membership_list* d_church_memberships_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_church_membership_count;         /**< message list counter*/
int h_message_church_membership_output_type;   /**< message output type (single or optional)*/

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_Person;
size_t temp_scan_storage_bytes_Person;

void * d_temp_scan_storage_Household;
size_t temp_scan_storage_bytes_Household;

void * d_temp_scan_storage_HouseholdMembership;
size_t temp_scan_storage_bytes_HouseholdMembership;

void * d_temp_scan_storage_Church;
size_t temp_scan_storage_bytes_Church;

void * d_temp_scan_storage_ChurchMembership;
size_t temp_scan_storage_bytes_ChurchMembership;

void * d_temp_scan_storage_Transport;
size_t temp_scan_storage_bytes_Transport;


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

/** Person_init
 * Agent function prototype for init function of Person agent
 */
void Person_init(cudaStream_t &stream);

/** Household_hhupdate
 * Agent function prototype for hhupdate function of Household agent
 */
void Household_hhupdate(cudaStream_t &stream);

/** HouseholdMembership_hhinit
 * Agent function prototype for hhinit function of HouseholdMembership agent
 */
void HouseholdMembership_hhinit(cudaStream_t &stream);

/** Church_chuupdate
 * Agent function prototype for chuupdate function of Church agent
 */
void Church_chuupdate(cudaStream_t &stream);

/** ChurchMembership_chuinit
 * Agent function prototype for chuinit function of ChurchMembership agent
 */
void ChurchMembership_chuinit(cudaStream_t &stream);

/** Transport_trupdate
 * Agent function prototype for trupdate function of Transport agent
 */
void Transport_trupdate(cudaStream_t &stream);

  
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
    h_Persons_default_variable_step_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    h_Persons_default_variable_churchfreq_data_iteration = 0;
    h_Persons_default_variable_churchdur_data_iteration = 0;
    h_Persons_default_variable_transportuser_data_iteration = 0;
    h_Persons_default_variable_transportfreq_data_iteration = 0;
    h_Persons_default_variable_transportdur_data_iteration = 0;
    h_Persons_default_variable_household_data_iteration = 0;
    h_Persons_default_variable_church_data_iteration = 0;
    h_Persons_default_variable_busy_data_iteration = 0;
    h_Persons_default_variable_startstep_data_iteration = 0;
    h_Persons_s2_variable_id_data_iteration = 0;
    h_Persons_s2_variable_step_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    h_Persons_s2_variable_churchfreq_data_iteration = 0;
    h_Persons_s2_variable_churchdur_data_iteration = 0;
    h_Persons_s2_variable_transportuser_data_iteration = 0;
    h_Persons_s2_variable_transportfreq_data_iteration = 0;
    h_Persons_s2_variable_transportdur_data_iteration = 0;
    h_Persons_s2_variable_household_data_iteration = 0;
    h_Persons_s2_variable_church_data_iteration = 0;
    h_Persons_s2_variable_busy_data_iteration = 0;
    h_Persons_s2_variable_startstep_data_iteration = 0;
    h_Households_hhdefault_variable_id_data_iteration = 0;
    h_Households_hhdefault_variable_step_data_iteration = 0;
    h_Households_hhdefault_variable_size_data_iteration = 0;
    h_Households_hhdefault_variable_people_data_iteration = 0;
    h_Households_hhdefault_variable_churchgoing_data_iteration = 0;
    h_Households_hhdefault_variable_churchfreq_data_iteration = 0;
    h_Households_hhdefault_variable_adults_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = 0;
    h_Churchs_chudefault_variable_id_data_iteration = 0;
    h_Churchs_chudefault_variable_step_data_iteration = 0;
    h_Churchs_chudefault_variable_size_data_iteration = 0;
    h_Churchs_chudefault_variable_duration_data_iteration = 0;
    h_Churchs_chudefault_variable_households_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration = 0;
    h_Transports_trdefault_variable_id_data_iteration = 0;
    h_Transports_trdefault_variable_step_data_iteration = 0;
    h_Transports_trdefault_variable_duration_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_Person_SoA_size = sizeof(xmachine_memory_Person_list);
	h_Persons_default = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);
	h_Persons_s2 = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);
	int xmachine_Household_SoA_size = sizeof(xmachine_memory_Household_list);
	h_Households_hhdefault = (xmachine_memory_Household_list*)malloc(xmachine_Household_SoA_size);
	int xmachine_HouseholdMembership_SoA_size = sizeof(xmachine_memory_HouseholdMembership_list);
	h_HouseholdMemberships_hhmembershipdefault = (xmachine_memory_HouseholdMembership_list*)malloc(xmachine_HouseholdMembership_SoA_size);
	int xmachine_Church_SoA_size = sizeof(xmachine_memory_Church_list);
	h_Churchs_chudefault = (xmachine_memory_Church_list*)malloc(xmachine_Church_SoA_size);
	int xmachine_ChurchMembership_SoA_size = sizeof(xmachine_memory_ChurchMembership_list);
	h_ChurchMemberships_chumembershipdefault = (xmachine_memory_ChurchMembership_list*)malloc(xmachine_ChurchMembership_SoA_size);
	int xmachine_Transport_SoA_size = sizeof(xmachine_memory_Transport_list);
	h_Transports_trdefault = (xmachine_memory_Transport_list*)malloc(xmachine_Transport_SoA_size);

	/* Message memory allocation (CPU) */
	int message_household_membership_SoA_size = sizeof(xmachine_message_household_membership_list);
	h_household_memberships = (xmachine_message_household_membership_list*)malloc(message_household_membership_SoA_size);
	int message_church_membership_SoA_size = sizeof(xmachine_message_church_membership_list);
	h_church_memberships = (xmachine_message_church_membership_list*)malloc(message_church_membership_SoA_size);
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs
    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_Persons_default, &h_xmachine_memory_Person_default_count, h_Households_hhdefault, &h_xmachine_memory_Household_hhdefault_count, h_HouseholdMemberships_hhmembershipdefault, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, h_Churchs_chudefault, &h_xmachine_memory_Church_chudefault_count, h_ChurchMemberships_chumembershipdefault, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, h_Transports_trdefault, &h_xmachine_memory_Transport_trdefault_count);
	

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
    
	/* Household Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Households, xmachine_Household_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Households_swap, xmachine_Household_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Households_new, xmachine_Household_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Household_keys, xmachine_memory_Household_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Household_values, xmachine_memory_Household_MAX* sizeof(uint)));
	/* hhdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Households_hhdefault, xmachine_Household_SoA_size));
	gpuErrchk( cudaMemcpy( d_Households_hhdefault, h_Households_hhdefault, xmachine_Household_SoA_size, cudaMemcpyHostToDevice));
    
	/* HouseholdMembership Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_HouseholdMemberships, xmachine_HouseholdMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_HouseholdMemberships_swap, xmachine_HouseholdMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_HouseholdMemberships_new, xmachine_HouseholdMembership_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_HouseholdMembership_keys, xmachine_memory_HouseholdMembership_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_HouseholdMembership_values, xmachine_memory_HouseholdMembership_MAX* sizeof(uint)));
	/* hhmembershipdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_HouseholdMemberships_hhmembershipdefault, xmachine_HouseholdMembership_SoA_size));
	gpuErrchk( cudaMemcpy( d_HouseholdMemberships_hhmembershipdefault, h_HouseholdMemberships_hhmembershipdefault, xmachine_HouseholdMembership_SoA_size, cudaMemcpyHostToDevice));
    
	/* Church Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Churchs, xmachine_Church_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Churchs_swap, xmachine_Church_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Churchs_new, xmachine_Church_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Church_keys, xmachine_memory_Church_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Church_values, xmachine_memory_Church_MAX* sizeof(uint)));
	/* chudefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Churchs_chudefault, xmachine_Church_SoA_size));
	gpuErrchk( cudaMemcpy( d_Churchs_chudefault, h_Churchs_chudefault, xmachine_Church_SoA_size, cudaMemcpyHostToDevice));
    
	/* ChurchMembership Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_ChurchMemberships, xmachine_ChurchMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_ChurchMemberships_swap, xmachine_ChurchMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_ChurchMemberships_new, xmachine_ChurchMembership_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_ChurchMembership_keys, xmachine_memory_ChurchMembership_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_ChurchMembership_values, xmachine_memory_ChurchMembership_MAX* sizeof(uint)));
	/* chumembershipdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_ChurchMemberships_chumembershipdefault, xmachine_ChurchMembership_SoA_size));
	gpuErrchk( cudaMemcpy( d_ChurchMemberships_chumembershipdefault, h_ChurchMemberships_chumembershipdefault, xmachine_ChurchMembership_SoA_size, cudaMemcpyHostToDevice));
    
	/* Transport Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Transports, xmachine_Transport_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Transports_swap, xmachine_Transport_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Transports_new, xmachine_Transport_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Transport_keys, xmachine_memory_Transport_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Transport_values, xmachine_memory_Transport_MAX* sizeof(uint)));
	/* trdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Transports_trdefault, xmachine_Transport_SoA_size));
	gpuErrchk( cudaMemcpy( d_Transports_trdefault, h_Transports_trdefault, xmachine_Transport_SoA_size, cudaMemcpyHostToDevice));
    
	/* household_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_household_memberships, message_household_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_household_memberships_swap, message_household_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_household_memberships, h_household_memberships, message_household_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* church_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_church_memberships, message_church_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_church_memberships_swap, message_church_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_church_memberships, h_church_memberships, message_church_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
		
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
    
    d_temp_scan_storage_Household = nullptr;
    temp_scan_storage_bytes_Household = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Household, 
        temp_scan_storage_bytes_Household, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Household_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Household, temp_scan_storage_bytes_Household));
    
    d_temp_scan_storage_HouseholdMembership = nullptr;
    temp_scan_storage_bytes_HouseholdMembership = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_HouseholdMembership, 
        temp_scan_storage_bytes_HouseholdMembership, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_HouseholdMembership_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_HouseholdMembership, temp_scan_storage_bytes_HouseholdMembership));
    
    d_temp_scan_storage_Church = nullptr;
    temp_scan_storage_bytes_Church = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Church, 
        temp_scan_storage_bytes_Church, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Church_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Church, temp_scan_storage_bytes_Church));
    
    d_temp_scan_storage_ChurchMembership = nullptr;
    temp_scan_storage_bytes_ChurchMembership = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_ChurchMembership, 
        temp_scan_storage_bytes_ChurchMembership, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_ChurchMembership_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_ChurchMembership, temp_scan_storage_bytes_ChurchMembership));
    
    d_temp_scan_storage_Transport = nullptr;
    temp_scan_storage_bytes_Transport = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Transport, 
        temp_scan_storage_bytes_Transport, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Transport_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Transport, temp_scan_storage_bytes_Transport));
    

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
    generateAgentsInit();
    PROFILE_PUSH_RANGE("generateAgentsInit");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: generateAgentsInit = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_Person_default_count: %u\n",get_agent_Person_default_count());
	
		printf("Init agent_Person_s2_count: %u\n",get_agent_Person_s2_count());
	
		printf("Init agent_Household_hhdefault_count: %u\n",get_agent_Household_hhdefault_count());
	
		printf("Init agent_HouseholdMembership_hhmembershipdefault_count: %u\n",get_agent_HouseholdMembership_hhmembershipdefault_count());
	
		printf("Init agent_Church_chudefault_count: %u\n",get_agent_Church_chudefault_count());
	
		printf("Init agent_ChurchMembership_chumembershipdefault_count: %u\n",get_agent_ChurchMembership_chumembershipdefault_count());
	
		printf("Init agent_Transport_trdefault_count: %u\n",get_agent_Transport_trdefault_count());
	
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

void sort_Households_hhdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Household_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Household_hhdefault_count); 
	gridSize = (h_xmachine_memory_Household_hhdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Household_keys, d_xmachine_memory_Household_values, d_Households_hhdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Household_keys),  thrust::device_pointer_cast(d_xmachine_memory_Household_keys) + h_xmachine_memory_Household_hhdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_Household_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Household_agents, no_sm, h_xmachine_memory_Household_hhdefault_count); 
	gridSize = (h_xmachine_memory_Household_hhdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Household_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Household_values, d_Households_hhdefault, d_Households_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Household_list* d_Households_temp = d_Households_hhdefault;
	d_Households_hhdefault = d_Households_swap;
	d_Households_swap = d_Households_temp;	
}

void sort_HouseholdMemberships_hhmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_HouseholdMembership_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count); 
	gridSize = (h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_HouseholdMembership_keys, d_xmachine_memory_HouseholdMembership_values, d_HouseholdMemberships_hhmembershipdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_HouseholdMembership_keys),  thrust::device_pointer_cast(d_xmachine_memory_HouseholdMembership_keys) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_HouseholdMembership_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_HouseholdMembership_agents, no_sm, h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count); 
	gridSize = (h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_HouseholdMembership_agents<<<gridSize, blockSize>>>(d_xmachine_memory_HouseholdMembership_values, d_HouseholdMemberships_hhmembershipdefault, d_HouseholdMemberships_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_temp = d_HouseholdMemberships_hhmembershipdefault;
	d_HouseholdMemberships_hhmembershipdefault = d_HouseholdMemberships_swap;
	d_HouseholdMemberships_swap = d_HouseholdMemberships_temp;	
}

void sort_Churchs_chudefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Church_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Church_chudefault_count); 
	gridSize = (h_xmachine_memory_Church_chudefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Church_keys, d_xmachine_memory_Church_values, d_Churchs_chudefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Church_keys),  thrust::device_pointer_cast(d_xmachine_memory_Church_keys) + h_xmachine_memory_Church_chudefault_count,  thrust::device_pointer_cast(d_xmachine_memory_Church_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Church_agents, no_sm, h_xmachine_memory_Church_chudefault_count); 
	gridSize = (h_xmachine_memory_Church_chudefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Church_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Church_values, d_Churchs_chudefault, d_Churchs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Church_list* d_Churchs_temp = d_Churchs_chudefault;
	d_Churchs_chudefault = d_Churchs_swap;
	d_Churchs_swap = d_Churchs_temp;	
}

void sort_ChurchMemberships_chumembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_ChurchMembership_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_ChurchMembership_chumembershipdefault_count); 
	gridSize = (h_xmachine_memory_ChurchMembership_chumembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_ChurchMembership_keys, d_xmachine_memory_ChurchMembership_values, d_ChurchMemberships_chumembershipdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_ChurchMembership_keys),  thrust::device_pointer_cast(d_xmachine_memory_ChurchMembership_keys) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_ChurchMembership_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_ChurchMembership_agents, no_sm, h_xmachine_memory_ChurchMembership_chumembershipdefault_count); 
	gridSize = (h_xmachine_memory_ChurchMembership_chumembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_ChurchMembership_agents<<<gridSize, blockSize>>>(d_xmachine_memory_ChurchMembership_values, d_ChurchMemberships_chumembershipdefault, d_ChurchMemberships_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_ChurchMembership_list* d_ChurchMemberships_temp = d_ChurchMemberships_chumembershipdefault;
	d_ChurchMemberships_chumembershipdefault = d_ChurchMemberships_swap;
	d_ChurchMemberships_swap = d_ChurchMemberships_temp;	
}

void sort_Transports_trdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Transport_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Transport_trdefault_count); 
	gridSize = (h_xmachine_memory_Transport_trdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Transport_keys, d_xmachine_memory_Transport_values, d_Transports_trdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Transport_keys),  thrust::device_pointer_cast(d_xmachine_memory_Transport_keys) + h_xmachine_memory_Transport_trdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_Transport_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Transport_agents, no_sm, h_xmachine_memory_Transport_trdefault_count); 
	gridSize = (h_xmachine_memory_Transport_trdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Transport_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Transport_values, d_Transports_trdefault, d_Transports_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Transport_list* d_Transports_temp = d_Transports_trdefault;
	d_Transports_trdefault = d_Transports_swap;
	d_Transports_swap = d_Transports_temp;	
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
	
	/* Household Agent variables */
	gpuErrchk(cudaFree(d_Households));
	gpuErrchk(cudaFree(d_Households_swap));
	gpuErrchk(cudaFree(d_Households_new));
	
	free( h_Households_hhdefault);
	gpuErrchk(cudaFree(d_Households_hhdefault));
	
	/* HouseholdMembership Agent variables */
	gpuErrchk(cudaFree(d_HouseholdMemberships));
	gpuErrchk(cudaFree(d_HouseholdMemberships_swap));
	gpuErrchk(cudaFree(d_HouseholdMemberships_new));
	
	free( h_HouseholdMemberships_hhmembershipdefault);
	gpuErrchk(cudaFree(d_HouseholdMemberships_hhmembershipdefault));
	
	/* Church Agent variables */
	gpuErrchk(cudaFree(d_Churchs));
	gpuErrchk(cudaFree(d_Churchs_swap));
	gpuErrchk(cudaFree(d_Churchs_new));
	
	free( h_Churchs_chudefault);
	gpuErrchk(cudaFree(d_Churchs_chudefault));
	
	/* ChurchMembership Agent variables */
	gpuErrchk(cudaFree(d_ChurchMemberships));
	gpuErrchk(cudaFree(d_ChurchMemberships_swap));
	gpuErrchk(cudaFree(d_ChurchMemberships_new));
	
	free( h_ChurchMemberships_chumembershipdefault);
	gpuErrchk(cudaFree(d_ChurchMemberships_chumembershipdefault));
	
	/* Transport Agent variables */
	gpuErrchk(cudaFree(d_Transports));
	gpuErrchk(cudaFree(d_Transports_swap));
	gpuErrchk(cudaFree(d_Transports_new));
	
	free( h_Transports_trdefault);
	gpuErrchk(cudaFree(d_Transports_trdefault));
	

	/* Message data free */
	
	/* household_membership Message variables */
	free( h_household_memberships);
	gpuErrchk(cudaFree(d_household_memberships));
	gpuErrchk(cudaFree(d_household_memberships_swap));
	
	/* church_membership Message variables */
	free( h_church_memberships);
	gpuErrchk(cudaFree(d_church_memberships));
	gpuErrchk(cudaFree(d_church_memberships_swap));
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Person));
    d_temp_scan_storage_Person = nullptr;
    temp_scan_storage_bytes_Person = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Household));
    d_temp_scan_storage_Household = nullptr;
    temp_scan_storage_bytes_Household = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_HouseholdMembership));
    d_temp_scan_storage_HouseholdMembership = nullptr;
    temp_scan_storage_bytes_HouseholdMembership = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Church));
    d_temp_scan_storage_Church = nullptr;
    temp_scan_storage_bytes_Church = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_ChurchMembership));
    d_temp_scan_storage_ChurchMembership = nullptr;
    temp_scan_storage_bytes_ChurchMembership = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Transport));
    d_temp_scan_storage_Transport = nullptr;
    temp_scan_storage_bytes_Transport = 0;
    
  
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
	h_message_household_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_household_membership_count, &h_message_household_membership_count, sizeof(int)));
	
	h_message_church_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_church_membership_count, &h_message_church_membership_count, sizeof(int)));
	
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("ChurchMembership_chuinit");
	ChurchMembership_chuinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: ChurchMembership_chuinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("HouseholdMembership_hhinit");
	HouseholdMembership_hhinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: HouseholdMembership_hhinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_init");
	Person_init(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_init = %f (ms)\n", instrument_milliseconds);
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
	
		printf("agent_Household_hhdefault_count: %u\n",get_agent_Household_hhdefault_count());
	
		printf("agent_HouseholdMembership_hhmembershipdefault_count: %u\n",get_agent_HouseholdMembership_hhmembershipdefault_count());
	
		printf("agent_Church_chudefault_count: %u\n",get_agent_Church_chudefault_count());
	
		printf("agent_ChurchMembership_chumembershipdefault_count: %u\n",get_agent_ChurchMembership_chumembershipdefault_count());
	
		printf("agent_Transport_trdefault_count: %u\n",get_agent_Transport_trdefault_count());
	
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
unsigned int h_env_MAX_AGE;
float h_env_STARTING_POPULATION;
float h_env_CHURCH_BETA0;
float h_env_CHURCH_BETA1;
unsigned int h_env_CHURCH_K1;
unsigned int h_env_CHURCH_K2;
unsigned int h_env_CHURCH_K3;
float h_env_CHURCH_P1;
float h_env_CHURCH_P2;
float h_env_CHURCH_PROB0;
float h_env_CHURCH_PROB1;
float h_env_CHURCH_PROB2;
float h_env_CHURCH_PROB3;
float h_env_CHURCH_PROB4;
float h_env_CHURCH_PROB5;
float h_env_CHURCH_PROB6;
float h_env_CHURCH_DURATION;
float h_env_TRANSPORT_BETA0;
float h_env_TRANSPORT_BETA1;
float h_env_TRANSPORT_FREQ0;
float h_env_TRANSPORT_FREQ2;
float h_env_TRANSPORT_DUR20;
float h_env_TRANSPORT_DUR45;
unsigned int h_env_TRANSPORT_SIZE;


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
void set_MAX_AGE(unsigned int* h_MAX_AGE){
    gpuErrchk(cudaMemcpyToSymbol(MAX_AGE, h_MAX_AGE, sizeof(unsigned int)));
    memcpy(&h_env_MAX_AGE, h_MAX_AGE,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_MAX_AGE(){
    return &h_env_MAX_AGE;
}



//constant setter
void set_STARTING_POPULATION(float* h_STARTING_POPULATION){
    gpuErrchk(cudaMemcpyToSymbol(STARTING_POPULATION, h_STARTING_POPULATION, sizeof(float)));
    memcpy(&h_env_STARTING_POPULATION, h_STARTING_POPULATION,sizeof(float));
}

//constant getter
const float* get_STARTING_POPULATION(){
    return &h_env_STARTING_POPULATION;
}



//constant setter
void set_CHURCH_BETA0(float* h_CHURCH_BETA0){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_BETA0, h_CHURCH_BETA0, sizeof(float)));
    memcpy(&h_env_CHURCH_BETA0, h_CHURCH_BETA0,sizeof(float));
}

//constant getter
const float* get_CHURCH_BETA0(){
    return &h_env_CHURCH_BETA0;
}



//constant setter
void set_CHURCH_BETA1(float* h_CHURCH_BETA1){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_BETA1, h_CHURCH_BETA1, sizeof(float)));
    memcpy(&h_env_CHURCH_BETA1, h_CHURCH_BETA1,sizeof(float));
}

//constant getter
const float* get_CHURCH_BETA1(){
    return &h_env_CHURCH_BETA1;
}



//constant setter
void set_CHURCH_K1(unsigned int* h_CHURCH_K1){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_K1, h_CHURCH_K1, sizeof(unsigned int)));
    memcpy(&h_env_CHURCH_K1, h_CHURCH_K1,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_CHURCH_K1(){
    return &h_env_CHURCH_K1;
}



//constant setter
void set_CHURCH_K2(unsigned int* h_CHURCH_K2){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_K2, h_CHURCH_K2, sizeof(unsigned int)));
    memcpy(&h_env_CHURCH_K2, h_CHURCH_K2,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_CHURCH_K2(){
    return &h_env_CHURCH_K2;
}



//constant setter
void set_CHURCH_K3(unsigned int* h_CHURCH_K3){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_K3, h_CHURCH_K3, sizeof(unsigned int)));
    memcpy(&h_env_CHURCH_K3, h_CHURCH_K3,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_CHURCH_K3(){
    return &h_env_CHURCH_K3;
}



//constant setter
void set_CHURCH_P1(float* h_CHURCH_P1){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_P1, h_CHURCH_P1, sizeof(float)));
    memcpy(&h_env_CHURCH_P1, h_CHURCH_P1,sizeof(float));
}

//constant getter
const float* get_CHURCH_P1(){
    return &h_env_CHURCH_P1;
}



//constant setter
void set_CHURCH_P2(float* h_CHURCH_P2){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_P2, h_CHURCH_P2, sizeof(float)));
    memcpy(&h_env_CHURCH_P2, h_CHURCH_P2,sizeof(float));
}

//constant getter
const float* get_CHURCH_P2(){
    return &h_env_CHURCH_P2;
}



//constant setter
void set_CHURCH_PROB0(float* h_CHURCH_PROB0){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB0, h_CHURCH_PROB0, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB0, h_CHURCH_PROB0,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB0(){
    return &h_env_CHURCH_PROB0;
}



//constant setter
void set_CHURCH_PROB1(float* h_CHURCH_PROB1){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB1, h_CHURCH_PROB1, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB1, h_CHURCH_PROB1,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB1(){
    return &h_env_CHURCH_PROB1;
}



//constant setter
void set_CHURCH_PROB2(float* h_CHURCH_PROB2){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB2, h_CHURCH_PROB2, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB2, h_CHURCH_PROB2,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB2(){
    return &h_env_CHURCH_PROB2;
}



//constant setter
void set_CHURCH_PROB3(float* h_CHURCH_PROB3){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB3, h_CHURCH_PROB3, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB3, h_CHURCH_PROB3,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB3(){
    return &h_env_CHURCH_PROB3;
}



//constant setter
void set_CHURCH_PROB4(float* h_CHURCH_PROB4){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB4, h_CHURCH_PROB4, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB4, h_CHURCH_PROB4,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB4(){
    return &h_env_CHURCH_PROB4;
}



//constant setter
void set_CHURCH_PROB5(float* h_CHURCH_PROB5){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB5, h_CHURCH_PROB5, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB5, h_CHURCH_PROB5,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB5(){
    return &h_env_CHURCH_PROB5;
}



//constant setter
void set_CHURCH_PROB6(float* h_CHURCH_PROB6){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_PROB6, h_CHURCH_PROB6, sizeof(float)));
    memcpy(&h_env_CHURCH_PROB6, h_CHURCH_PROB6,sizeof(float));
}

//constant getter
const float* get_CHURCH_PROB6(){
    return &h_env_CHURCH_PROB6;
}



//constant setter
void set_CHURCH_DURATION(float* h_CHURCH_DURATION){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_DURATION, h_CHURCH_DURATION, sizeof(float)));
    memcpy(&h_env_CHURCH_DURATION, h_CHURCH_DURATION,sizeof(float));
}

//constant getter
const float* get_CHURCH_DURATION(){
    return &h_env_CHURCH_DURATION;
}



//constant setter
void set_TRANSPORT_BETA0(float* h_TRANSPORT_BETA0){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_BETA0, h_TRANSPORT_BETA0, sizeof(float)));
    memcpy(&h_env_TRANSPORT_BETA0, h_TRANSPORT_BETA0,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_BETA0(){
    return &h_env_TRANSPORT_BETA0;
}



//constant setter
void set_TRANSPORT_BETA1(float* h_TRANSPORT_BETA1){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_BETA1, h_TRANSPORT_BETA1, sizeof(float)));
    memcpy(&h_env_TRANSPORT_BETA1, h_TRANSPORT_BETA1,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_BETA1(){
    return &h_env_TRANSPORT_BETA1;
}



//constant setter
void set_TRANSPORT_FREQ0(float* h_TRANSPORT_FREQ0){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_FREQ0, h_TRANSPORT_FREQ0, sizeof(float)));
    memcpy(&h_env_TRANSPORT_FREQ0, h_TRANSPORT_FREQ0,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_FREQ0(){
    return &h_env_TRANSPORT_FREQ0;
}



//constant setter
void set_TRANSPORT_FREQ2(float* h_TRANSPORT_FREQ2){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_FREQ2, h_TRANSPORT_FREQ2, sizeof(float)));
    memcpy(&h_env_TRANSPORT_FREQ2, h_TRANSPORT_FREQ2,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_FREQ2(){
    return &h_env_TRANSPORT_FREQ2;
}



//constant setter
void set_TRANSPORT_DUR20(float* h_TRANSPORT_DUR20){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_DUR20, h_TRANSPORT_DUR20, sizeof(float)));
    memcpy(&h_env_TRANSPORT_DUR20, h_TRANSPORT_DUR20,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_DUR20(){
    return &h_env_TRANSPORT_DUR20;
}



//constant setter
void set_TRANSPORT_DUR45(float* h_TRANSPORT_DUR45){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_DUR45, h_TRANSPORT_DUR45, sizeof(float)));
    memcpy(&h_env_TRANSPORT_DUR45, h_TRANSPORT_DUR45,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_DUR45(){
    return &h_env_TRANSPORT_DUR45;
}



//constant setter
void set_TRANSPORT_SIZE(unsigned int* h_TRANSPORT_SIZE){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_SIZE, h_TRANSPORT_SIZE, sizeof(unsigned int)));
    memcpy(&h_env_TRANSPORT_SIZE, h_TRANSPORT_SIZE,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_TRANSPORT_SIZE(){
    return &h_env_TRANSPORT_SIZE;
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

    
int get_agent_Household_MAX_count(){
    return xmachine_memory_Household_MAX;
}


int get_agent_Household_hhdefault_count(){
	//continuous agent
	return h_xmachine_memory_Household_hhdefault_count;
	
}

xmachine_memory_Household_list* get_device_Household_hhdefault_agents(){
	return d_Households_hhdefault;
}

xmachine_memory_Household_list* get_host_Household_hhdefault_agents(){
	return h_Households_hhdefault;
}

    
int get_agent_HouseholdMembership_MAX_count(){
    return xmachine_memory_HouseholdMembership_MAX;
}


int get_agent_HouseholdMembership_hhmembershipdefault_count(){
	//continuous agent
	return h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;
	
}

xmachine_memory_HouseholdMembership_list* get_device_HouseholdMembership_hhmembershipdefault_agents(){
	return d_HouseholdMemberships_hhmembershipdefault;
}

xmachine_memory_HouseholdMembership_list* get_host_HouseholdMembership_hhmembershipdefault_agents(){
	return h_HouseholdMemberships_hhmembershipdefault;
}

    
int get_agent_Church_MAX_count(){
    return xmachine_memory_Church_MAX;
}


int get_agent_Church_chudefault_count(){
	//continuous agent
	return h_xmachine_memory_Church_chudefault_count;
	
}

xmachine_memory_Church_list* get_device_Church_chudefault_agents(){
	return d_Churchs_chudefault;
}

xmachine_memory_Church_list* get_host_Church_chudefault_agents(){
	return h_Churchs_chudefault;
}

    
int get_agent_ChurchMembership_MAX_count(){
    return xmachine_memory_ChurchMembership_MAX;
}


int get_agent_ChurchMembership_chumembershipdefault_count(){
	//continuous agent
	return h_xmachine_memory_ChurchMembership_chumembershipdefault_count;
	
}

xmachine_memory_ChurchMembership_list* get_device_ChurchMembership_chumembershipdefault_agents(){
	return d_ChurchMemberships_chumembershipdefault;
}

xmachine_memory_ChurchMembership_list* get_host_ChurchMembership_chumembershipdefault_agents(){
	return h_ChurchMemberships_chumembershipdefault;
}

    
int get_agent_Transport_MAX_count(){
    return xmachine_memory_Transport_MAX;
}


int get_agent_Transport_trdefault_count(){
	//continuous agent
	return h_xmachine_memory_Transport_trdefault_count;
	
}

xmachine_memory_Transport_list* get_device_Transport_trdefault_agents(){
	return d_Transports_trdefault;
}

xmachine_memory_Transport_list* get_host_Transport_trdefault_agents(){
	return h_Transports_trdefault;
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

/** unsigned int get_Person_default_variable_step(unsigned int index)
 * Gets the value of the step variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Person_default_variable_step(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->step,
                    d_Persons_default->step,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access step for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_default_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Person_default_variable_churchfreq(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_churchfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->churchfreq,
                    d_Persons_default->churchfreq,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_churchfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->churchfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchfreq for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_default_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_Person_default_variable_churchdur(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_churchdur_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->churchdur,
                    d_Persons_default->churchdur,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_churchdur_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->churchdur[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchdur for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_transportuser(unsigned int index)
 * Gets the value of the transportuser variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportuser
 */
__host__ unsigned int get_Person_default_variable_transportuser(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transportuser_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transportuser,
                    d_Persons_default->transportuser,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transportuser_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transportuser[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportuser for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_transportfreq(unsigned int index)
 * Gets the value of the transportfreq variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportfreq
 */
__host__ int get_Person_default_variable_transportfreq(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transportfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transportfreq,
                    d_Persons_default->transportfreq,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transportfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transportfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportfreq for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ int get_Person_default_variable_transportdur(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transportdur_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transportdur,
                    d_Persons_default->transportdur,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transportdur_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transportdur[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportdur for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_household(unsigned int index)
 * Gets the value of the household variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household
 */
__host__ unsigned int get_Person_default_variable_household(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_household_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->household,
                    d_Persons_default->household,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_household_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->household[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access household for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ unsigned int get_Person_default_variable_church(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_church_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->church,
                    d_Persons_default->church,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_church_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->church[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access church for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_busy(unsigned int index)
 * Gets the value of the busy variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable busy
 */
__host__ unsigned int get_Person_default_variable_busy(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_busy_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->busy,
                    d_Persons_default->busy,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_busy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->busy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access busy for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_startstep(unsigned int index)
 * Gets the value of the startstep variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable startstep
 */
__host__ unsigned int get_Person_default_variable_startstep(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_startstep_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->startstep,
                    d_Persons_default->startstep,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_startstep_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->startstep[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access startstep for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_s2_variable_step(unsigned int index)
 * Gets the value of the step variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Person_s2_variable_step(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->step,
                    d_Persons_s2->step,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access step for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_s2_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Person_s2_variable_churchfreq(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_churchfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->churchfreq,
                    d_Persons_s2->churchfreq,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_churchfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->churchfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchfreq for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_s2_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_Person_s2_variable_churchdur(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_churchdur_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->churchdur,
                    d_Persons_s2->churchdur,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_churchdur_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->churchdur[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchdur for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_transportuser(unsigned int index)
 * Gets the value of the transportuser variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportuser
 */
__host__ unsigned int get_Person_s2_variable_transportuser(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transportuser_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transportuser,
                    d_Persons_s2->transportuser,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transportuser_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transportuser[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportuser for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_transportfreq(unsigned int index)
 * Gets the value of the transportfreq variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportfreq
 */
__host__ int get_Person_s2_variable_transportfreq(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transportfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transportfreq,
                    d_Persons_s2->transportfreq,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transportfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transportfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportfreq for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ int get_Person_s2_variable_transportdur(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transportdur_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transportdur,
                    d_Persons_s2->transportdur,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transportdur_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transportdur[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportdur for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_household(unsigned int index)
 * Gets the value of the household variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household
 */
__host__ unsigned int get_Person_s2_variable_household(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_household_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->household,
                    d_Persons_s2->household,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_household_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->household[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access household for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ unsigned int get_Person_s2_variable_church(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_church_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->church,
                    d_Persons_s2->church,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_church_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->church[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access church for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_busy(unsigned int index)
 * Gets the value of the busy variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable busy
 */
__host__ unsigned int get_Person_s2_variable_busy(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_busy_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->busy,
                    d_Persons_s2->busy,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_busy_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->busy[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access busy for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_startstep(unsigned int index)
 * Gets the value of the startstep variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable startstep
 */
__host__ unsigned int get_Person_s2_variable_startstep(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_startstep_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->startstep,
                    d_Persons_s2->startstep,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_startstep_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->startstep[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access startstep for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Household_hhdefault_variable_id(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->id,
                    d_Households_hhdefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_step(unsigned int index)
 * Gets the value of the step variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Household_hhdefault_variable_step(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->step,
                    d_Households_hhdefault->step,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access step for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_size(unsigned int index)
 * Gets the value of the size variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_Household_hhdefault_variable_size(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_size_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->size,
                    d_Households_hhdefault->size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access size for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Household_hhdefault_variable_people(unsigned int index, unsigned int element)
 * Gets the element-th value of the people variable array of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable people
 */
__host__ int get_Household_hhdefault_variable_people(unsigned int index, unsigned int element){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int numElements = 32;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_people_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_Households_hhdefault->people + (e * xmachine_memory_Household_MAX),
                        d_Households_hhdefault->people + (e * xmachine_memory_Household_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_Households_hhdefault_variable_people_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->people[index + (element * xmachine_memory_Household_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of people for the %u th member of Household_hhdefault. count is %u at iteration %u\n", element, index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_churchgoing(unsigned int index)
 * Gets the value of the churchgoing variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchgoing
 */
__host__ unsigned int get_Household_hhdefault_variable_churchgoing(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_churchgoing_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->churchgoing,
                    d_Households_hhdefault->churchgoing,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_churchgoing_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->churchgoing[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchgoing for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Household_hhdefault_variable_churchfreq(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_churchfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->churchfreq,
                    d_Households_hhdefault->churchfreq,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_churchfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->churchfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchfreq for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_adults(unsigned int index)
 * Gets the value of the adults variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable adults
 */
__host__ unsigned int get_Household_hhdefault_variable_adults(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_adults_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->adults,
                    d_Households_hhdefault->adults,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_adults_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->adults[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access adults for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_id(unsigned int index)
 * Gets the value of the household_id variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_id
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_id(unsigned int index){
    unsigned int count = get_agent_HouseholdMembership_hhmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_HouseholdMemberships_hhmembershipdefault->household_id,
                    d_HouseholdMemberships_hhmembershipdefault->household_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_HouseholdMemberships_hhmembershipdefault->household_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access household_id for the %u th member of HouseholdMembership_hhmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_person_id(unsigned int index){
    unsigned int count = get_agent_HouseholdMembership_hhmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_HouseholdMemberships_hhmembershipdefault->person_id,
                    d_HouseholdMemberships_hhmembershipdefault->person_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_HouseholdMemberships_hhmembershipdefault->person_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access person_id for the %u th member of HouseholdMembership_hhmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchgoing(unsigned int index)
 * Gets the value of the churchgoing variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchgoing
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchgoing(unsigned int index){
    unsigned int count = get_agent_HouseholdMembership_hhmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_HouseholdMemberships_hhmembershipdefault->churchgoing,
                    d_HouseholdMemberships_hhmembershipdefault->churchgoing,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_HouseholdMemberships_hhmembershipdefault->churchgoing[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchgoing for the %u th member of HouseholdMembership_hhmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchfreq(unsigned int index){
    unsigned int count = get_agent_HouseholdMembership_hhmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_HouseholdMemberships_hhmembershipdefault->churchfreq,
                    d_HouseholdMemberships_hhmembershipdefault->churchfreq,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_HouseholdMemberships_hhmembershipdefault->churchfreq[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchfreq for the %u th member of HouseholdMembership_hhmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Church_chudefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Church_chudefault_variable_id(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->id,
                    d_Churchs_chudefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Church_chudefault_variable_step(unsigned int index)
 * Gets the value of the step variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Church_chudefault_variable_step(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->step,
                    d_Churchs_chudefault->step,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access step for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Church_chudefault_variable_size(unsigned int index)
 * Gets the value of the size variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_Church_chudefault_variable_size(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_size_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->size,
                    d_Churchs_chudefault->size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access size for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Church_chudefault_variable_duration(unsigned int index)
 * Gets the value of the duration variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable duration
 */
__host__ float get_Church_chudefault_variable_duration(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_duration_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->duration,
                    d_Churchs_chudefault->duration,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_duration_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->duration[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access duration for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Church_chudefault_variable_households(unsigned int index, unsigned int element)
 * Gets the element-th value of the households variable array of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable households
 */
__host__ int get_Church_chudefault_variable_households(unsigned int index, unsigned int element){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int numElements = 128;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_households_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_Churchs_chudefault->households + (e * xmachine_memory_Church_MAX),
                        d_Churchs_chudefault->households + (e * xmachine_memory_Church_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_Churchs_chudefault_variable_households_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->households[index + (element * xmachine_memory_Church_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of households for the %u th member of Church_chudefault. count is %u at iteration %u\n", element, index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_ChurchMembership_chumembershipdefault_variable_church_id(unsigned int index)
 * Gets the value of the church_id variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church_id
 */
__host__ unsigned int get_ChurchMembership_chumembershipdefault_variable_church_id(unsigned int index){
    unsigned int count = get_agent_ChurchMembership_chumembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_ChurchMemberships_chumembershipdefault->church_id,
                    d_ChurchMemberships_chumembershipdefault->church_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_ChurchMemberships_chumembershipdefault->church_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access church_id for the %u th member of ChurchMembership_chumembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_ChurchMembership_chumembershipdefault_variable_household_id(unsigned int index)
 * Gets the value of the household_id variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_id
 */
__host__ unsigned int get_ChurchMembership_chumembershipdefault_variable_household_id(unsigned int index){
    unsigned int count = get_agent_ChurchMembership_chumembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_ChurchMemberships_chumembershipdefault->household_id,
                    d_ChurchMemberships_chumembershipdefault->household_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_ChurchMemberships_chumembershipdefault->household_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access household_id for the %u th member of ChurchMembership_chumembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_ChurchMembership_chumembershipdefault_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_ChurchMembership_chumembershipdefault_variable_churchdur(unsigned int index){
    unsigned int count = get_agent_ChurchMembership_chumembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_ChurchMemberships_chumembershipdefault->churchdur,
                    d_ChurchMemberships_chumembershipdefault->churchdur,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_ChurchMemberships_chumembershipdefault->churchdur[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchdur for the %u th member of ChurchMembership_chumembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Transport_trdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Transport_trdefault_variable_id(unsigned int index){
    unsigned int count = get_agent_Transport_trdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Transports_trdefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Transports_trdefault->id,
                    d_Transports_trdefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Transports_trdefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Transports_trdefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Transport_trdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Transport_trdefault_variable_step(unsigned int index)
 * Gets the value of the step variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Transport_trdefault_variable_step(unsigned int index){
    unsigned int count = get_agent_Transport_trdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Transports_trdefault_variable_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Transports_trdefault->step,
                    d_Transports_trdefault->step,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Transports_trdefault_variable_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Transports_trdefault->step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access step for the %u th member of Transport_trdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Transport_trdefault_variable_duration(unsigned int index)
 * Gets the value of the duration variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable duration
 */
__host__ unsigned int get_Transport_trdefault_variable_duration(unsigned int index){
    unsigned int count = get_agent_Transport_trdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Transports_trdefault_variable_duration_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Transports_trdefault->duration,
                    d_Transports_trdefault->duration,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Transports_trdefault_variable_duration_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Transports_trdefault->duration[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access duration for the %u th member of Transport_trdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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
 
		gpuErrchk(cudaMemcpy(d_dst->step, &h_agent->step, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, &h_agent->age, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, &h_agent->gender, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, &h_agent->householdsize, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, &h_agent->churchfreq, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, &h_agent->churchdur, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportuser, &h_agent->transportuser, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportfreq, &h_agent->transportfreq, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportdur, &h_agent->transportdur, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household, &h_agent->household, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->church, &h_agent->church, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->busy, &h_agent->busy, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->startstep, &h_agent->startstep, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->step, h_src->step, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, h_src->age, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, h_src->gender, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, h_src->householdsize, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, h_src->churchfreq, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, h_src->churchdur, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportuser, h_src->transportuser, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportfreq, h_src->transportfreq, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportdur, h_src->transportdur, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household, h_src->household, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->church, h_src->church, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->busy, h_src->busy, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->startstep, h_src->startstep, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Household_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Household_hostToDevice(xmachine_memory_Household_list * d_dst, xmachine_memory_Household * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, &h_agent->step, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 32; i++){
		gpuErrchk(cudaMemcpy(d_dst->people + (i * xmachine_memory_Household_MAX), h_agent->people + i, sizeof(int), cudaMemcpyHostToDevice));
    }
 
		gpuErrchk(cudaMemcpy(d_dst->churchgoing, &h_agent->churchgoing, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, &h_agent->churchfreq, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->adults, &h_agent->adults, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_Household_hostToDevice(xmachine_memory_Household_list * d_dst, xmachine_memory_Household_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, h_src->step, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 32; i++){
			gpuErrchk(cudaMemcpy(d_dst->people + (i * xmachine_memory_Household_MAX), h_src->people + (i * xmachine_memory_Household_MAX), count * sizeof(int), cudaMemcpyHostToDevice));
        }

 
		gpuErrchk(cudaMemcpy(d_dst->churchgoing, h_src->churchgoing, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, h_src->churchfreq, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->adults, h_src->adults, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_HouseholdMembership_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_HouseholdMembership_hostToDevice(xmachine_memory_HouseholdMembership_list * d_dst, xmachine_memory_HouseholdMembership * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->household_id, &h_agent->household_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->person_id, &h_agent->person_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchgoing, &h_agent->churchgoing, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, &h_agent->churchfreq, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_HouseholdMembership_hostToDevice(xmachine_memory_HouseholdMembership_list * d_dst, xmachine_memory_HouseholdMembership_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->household_id, h_src->household_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->person_id, h_src->person_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchgoing, h_src->churchgoing, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, h_src->churchfreq, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Church_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Church_hostToDevice(xmachine_memory_Church_list * d_dst, xmachine_memory_Church * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, &h_agent->step, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->duration, &h_agent->duration, sizeof(float), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 128; i++){
		gpuErrchk(cudaMemcpy(d_dst->households + (i * xmachine_memory_Church_MAX), h_agent->households + i, sizeof(int), cudaMemcpyHostToDevice));
    }

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
void copy_partial_xmachine_memory_Church_hostToDevice(xmachine_memory_Church_list * d_dst, xmachine_memory_Church_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, h_src->step, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->duration, h_src->duration, count * sizeof(float), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 128; i++){
			gpuErrchk(cudaMemcpy(d_dst->households + (i * xmachine_memory_Church_MAX), h_src->households + (i * xmachine_memory_Church_MAX), count * sizeof(int), cudaMemcpyHostToDevice));
        }


    }
}


/* copy_single_xmachine_memory_ChurchMembership_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_ChurchMembership_hostToDevice(xmachine_memory_ChurchMembership_list * d_dst, xmachine_memory_ChurchMembership * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->church_id, &h_agent->church_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household_id, &h_agent->household_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, &h_agent->churchdur, sizeof(float), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_ChurchMembership_hostToDevice(xmachine_memory_ChurchMembership_list * d_dst, xmachine_memory_ChurchMembership_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->church_id, h_src->church_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household_id, h_src->household_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, h_src->churchdur, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Transport_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Transport_hostToDevice(xmachine_memory_Transport_list * d_dst, xmachine_memory_Transport * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, &h_agent->step, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->duration, &h_agent->duration, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_Transport_hostToDevice(xmachine_memory_Transport_list * d_dst, xmachine_memory_Transport_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->step, h_src->step, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->duration, h_src->duration, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
			 
			dst->step[i] = src[i]->step;
			 
			dst->age[i] = src[i]->age;
			 
			dst->gender[i] = src[i]->gender;
			 
			dst->householdsize[i] = src[i]->householdsize;
			 
			dst->churchfreq[i] = src[i]->churchfreq;
			 
			dst->churchdur[i] = src[i]->churchdur;
			 
			dst->transportuser[i] = src[i]->transportuser;
			 
			dst->transportfreq[i] = src[i]->transportfreq;
			 
			dst->transportdur[i] = src[i]->transportdur;
			 
			dst->household[i] = src[i]->household;
			 
			dst->church[i] = src[i]->church;
			 
			dst->busy[i] = src[i]->busy;
			 
			dst->startstep[i] = src[i]->startstep;
			
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
    h_Persons_default_variable_step_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    h_Persons_default_variable_churchfreq_data_iteration = 0;
    h_Persons_default_variable_churchdur_data_iteration = 0;
    h_Persons_default_variable_transportuser_data_iteration = 0;
    h_Persons_default_variable_transportfreq_data_iteration = 0;
    h_Persons_default_variable_transportdur_data_iteration = 0;
    h_Persons_default_variable_household_data_iteration = 0;
    h_Persons_default_variable_church_data_iteration = 0;
    h_Persons_default_variable_busy_data_iteration = 0;
    h_Persons_default_variable_startstep_data_iteration = 0;
    

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
        h_Persons_default_variable_step_data_iteration = 0;
        h_Persons_default_variable_age_data_iteration = 0;
        h_Persons_default_variable_gender_data_iteration = 0;
        h_Persons_default_variable_householdsize_data_iteration = 0;
        h_Persons_default_variable_churchfreq_data_iteration = 0;
        h_Persons_default_variable_churchdur_data_iteration = 0;
        h_Persons_default_variable_transportuser_data_iteration = 0;
        h_Persons_default_variable_transportfreq_data_iteration = 0;
        h_Persons_default_variable_transportdur_data_iteration = 0;
        h_Persons_default_variable_household_data_iteration = 0;
        h_Persons_default_variable_church_data_iteration = 0;
        h_Persons_default_variable_busy_data_iteration = 0;
        h_Persons_default_variable_startstep_data_iteration = 0;
        

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
    h_Persons_s2_variable_step_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    h_Persons_s2_variable_churchfreq_data_iteration = 0;
    h_Persons_s2_variable_churchdur_data_iteration = 0;
    h_Persons_s2_variable_transportuser_data_iteration = 0;
    h_Persons_s2_variable_transportfreq_data_iteration = 0;
    h_Persons_s2_variable_transportdur_data_iteration = 0;
    h_Persons_s2_variable_household_data_iteration = 0;
    h_Persons_s2_variable_church_data_iteration = 0;
    h_Persons_s2_variable_busy_data_iteration = 0;
    h_Persons_s2_variable_startstep_data_iteration = 0;
    

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
        h_Persons_s2_variable_step_data_iteration = 0;
        h_Persons_s2_variable_age_data_iteration = 0;
        h_Persons_s2_variable_gender_data_iteration = 0;
        h_Persons_s2_variable_householdsize_data_iteration = 0;
        h_Persons_s2_variable_churchfreq_data_iteration = 0;
        h_Persons_s2_variable_churchdur_data_iteration = 0;
        h_Persons_s2_variable_transportuser_data_iteration = 0;
        h_Persons_s2_variable_transportfreq_data_iteration = 0;
        h_Persons_s2_variable_transportdur_data_iteration = 0;
        h_Persons_s2_variable_household_data_iteration = 0;
        h_Persons_s2_variable_church_data_iteration = 0;
        h_Persons_s2_variable_busy_data_iteration = 0;
        h_Persons_s2_variable_startstep_data_iteration = 0;
        

	}
}

xmachine_memory_Household* h_allocate_agent_Household(){
	xmachine_memory_Household* agent = (xmachine_memory_Household*)malloc(sizeof(xmachine_memory_Household));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Household));
	// Agent variable arrays must be allocated
    agent->people = (int*)malloc(32 * sizeof(int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 32; index++){
		agent->people[index] = -1;
	}
	return agent;
}
void h_free_agent_Household(xmachine_memory_Household** agent){

    free((*agent)->people);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Household** h_allocate_agent_Household_array(unsigned int count){
	xmachine_memory_Household ** agents = (xmachine_memory_Household**)malloc(count * sizeof(xmachine_memory_Household*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Household();
	}
	return agents;
}
void h_free_agent_Household_array(xmachine_memory_Household*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Household(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Household_AoS_to_SoA(xmachine_memory_Household_list * dst, xmachine_memory_Household** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->step[i] = src[i]->step;
			 
			dst->size[i] = src[i]->size;
			 
			for(unsigned int j = 0; j < 32; j++){
				dst->people[(j * xmachine_memory_Household_MAX) + i] = src[i]->people[j];
			}
			 
			dst->churchgoing[i] = src[i]->churchgoing;
			 
			dst->churchfreq[i] = src[i]->churchfreq;
			 
			dst->adults[i] = src[i]->adults;
			
		}
	}
}


void h_add_agent_Household_hhdefault(xmachine_memory_Household* agent){
	if (h_xmachine_memory_Household_count + 1 > xmachine_memory_Household_MAX){
		printf("Error: Buffer size of Household agents in state hhdefault will be exceeded by h_add_agent_Household_hhdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Household_hostToDevice(d_Households_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Household_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Household_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Households_hhdefault, d_Households_new, h_xmachine_memory_Household_hhdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Household_hhdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Household_hhdefault_count, &h_xmachine_memory_Household_hhdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Households_hhdefault_variable_id_data_iteration = 0;
    h_Households_hhdefault_variable_step_data_iteration = 0;
    h_Households_hhdefault_variable_size_data_iteration = 0;
    h_Households_hhdefault_variable_people_data_iteration = 0;
    h_Households_hhdefault_variable_churchgoing_data_iteration = 0;
    h_Households_hhdefault_variable_churchfreq_data_iteration = 0;
    h_Households_hhdefault_variable_adults_data_iteration = 0;
    

}
void h_add_agents_Household_hhdefault(xmachine_memory_Household** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Household_count + count > xmachine_memory_Household_MAX){
			printf("Error: Buffer size of Household agents in state hhdefault will be exceeded by h_add_agents_Household_hhdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Household_AoS_to_SoA(h_Households_hhdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Household_hostToDevice(d_Households_new, h_Households_hhdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Household_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Household_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Households_hhdefault, d_Households_new, h_xmachine_memory_Household_hhdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Household_hhdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Household_hhdefault_count, &h_xmachine_memory_Household_hhdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Households_hhdefault_variable_id_data_iteration = 0;
        h_Households_hhdefault_variable_step_data_iteration = 0;
        h_Households_hhdefault_variable_size_data_iteration = 0;
        h_Households_hhdefault_variable_people_data_iteration = 0;
        h_Households_hhdefault_variable_churchgoing_data_iteration = 0;
        h_Households_hhdefault_variable_churchfreq_data_iteration = 0;
        h_Households_hhdefault_variable_adults_data_iteration = 0;
        

	}
}

xmachine_memory_HouseholdMembership* h_allocate_agent_HouseholdMembership(){
	xmachine_memory_HouseholdMembership* agent = (xmachine_memory_HouseholdMembership*)malloc(sizeof(xmachine_memory_HouseholdMembership));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_HouseholdMembership));

	return agent;
}
void h_free_agent_HouseholdMembership(xmachine_memory_HouseholdMembership** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_HouseholdMembership** h_allocate_agent_HouseholdMembership_array(unsigned int count){
	xmachine_memory_HouseholdMembership ** agents = (xmachine_memory_HouseholdMembership**)malloc(count * sizeof(xmachine_memory_HouseholdMembership*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_HouseholdMembership();
	}
	return agents;
}
void h_free_agent_HouseholdMembership_array(xmachine_memory_HouseholdMembership*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_HouseholdMembership(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_HouseholdMembership_AoS_to_SoA(xmachine_memory_HouseholdMembership_list * dst, xmachine_memory_HouseholdMembership** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->household_id[i] = src[i]->household_id;
			 
			dst->person_id[i] = src[i]->person_id;
			 
			dst->churchgoing[i] = src[i]->churchgoing;
			 
			dst->churchfreq[i] = src[i]->churchfreq;
			
		}
	}
}


void h_add_agent_HouseholdMembership_hhmembershipdefault(xmachine_memory_HouseholdMembership* agent){
	if (h_xmachine_memory_HouseholdMembership_count + 1 > xmachine_memory_HouseholdMembership_MAX){
		printf("Error: Buffer size of HouseholdMembership agents in state hhmembershipdefault will be exceeded by h_add_agent_HouseholdMembership_hhmembershipdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_HouseholdMembership_hostToDevice(d_HouseholdMemberships_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_HouseholdMembership_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_HouseholdMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_HouseholdMemberships_hhmembershipdefault, d_HouseholdMemberships_new, h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = 0;
    

}
void h_add_agents_HouseholdMembership_hhmembershipdefault(xmachine_memory_HouseholdMembership** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_HouseholdMembership_count + count > xmachine_memory_HouseholdMembership_MAX){
			printf("Error: Buffer size of HouseholdMembership agents in state hhmembershipdefault will be exceeded by h_add_agents_HouseholdMembership_hhmembershipdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_HouseholdMembership_AoS_to_SoA(h_HouseholdMemberships_hhmembershipdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_HouseholdMembership_hostToDevice(d_HouseholdMemberships_new, h_HouseholdMemberships_hhmembershipdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_HouseholdMembership_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_HouseholdMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_HouseholdMemberships_hhmembershipdefault, d_HouseholdMemberships_new, h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration = 0;
        h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration = 0;
        h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = 0;
        h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = 0;
        

	}
}

xmachine_memory_Church* h_allocate_agent_Church(){
	xmachine_memory_Church* agent = (xmachine_memory_Church*)malloc(sizeof(xmachine_memory_Church));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Church));
	// Agent variable arrays must be allocated
    agent->households = (int*)malloc(128 * sizeof(int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 128; index++){
		agent->households[index] = -1;
	}
	return agent;
}
void h_free_agent_Church(xmachine_memory_Church** agent){

    free((*agent)->households);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Church** h_allocate_agent_Church_array(unsigned int count){
	xmachine_memory_Church ** agents = (xmachine_memory_Church**)malloc(count * sizeof(xmachine_memory_Church*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Church();
	}
	return agents;
}
void h_free_agent_Church_array(xmachine_memory_Church*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Church(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Church_AoS_to_SoA(xmachine_memory_Church_list * dst, xmachine_memory_Church** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->step[i] = src[i]->step;
			 
			dst->size[i] = src[i]->size;
			 
			dst->duration[i] = src[i]->duration;
			 
			for(unsigned int j = 0; j < 128; j++){
				dst->households[(j * xmachine_memory_Church_MAX) + i] = src[i]->households[j];
			}
			
		}
	}
}


void h_add_agent_Church_chudefault(xmachine_memory_Church* agent){
	if (h_xmachine_memory_Church_count + 1 > xmachine_memory_Church_MAX){
		printf("Error: Buffer size of Church agents in state chudefault will be exceeded by h_add_agent_Church_chudefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Church_hostToDevice(d_Churchs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Church_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Church_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Churchs_chudefault, d_Churchs_new, h_xmachine_memory_Church_chudefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Church_chudefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Church_chudefault_count, &h_xmachine_memory_Church_chudefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Churchs_chudefault_variable_id_data_iteration = 0;
    h_Churchs_chudefault_variable_step_data_iteration = 0;
    h_Churchs_chudefault_variable_size_data_iteration = 0;
    h_Churchs_chudefault_variable_duration_data_iteration = 0;
    h_Churchs_chudefault_variable_households_data_iteration = 0;
    

}
void h_add_agents_Church_chudefault(xmachine_memory_Church** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Church_count + count > xmachine_memory_Church_MAX){
			printf("Error: Buffer size of Church agents in state chudefault will be exceeded by h_add_agents_Church_chudefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Church_AoS_to_SoA(h_Churchs_chudefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Church_hostToDevice(d_Churchs_new, h_Churchs_chudefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Church_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Church_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Churchs_chudefault, d_Churchs_new, h_xmachine_memory_Church_chudefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Church_chudefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Church_chudefault_count, &h_xmachine_memory_Church_chudefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Churchs_chudefault_variable_id_data_iteration = 0;
        h_Churchs_chudefault_variable_step_data_iteration = 0;
        h_Churchs_chudefault_variable_size_data_iteration = 0;
        h_Churchs_chudefault_variable_duration_data_iteration = 0;
        h_Churchs_chudefault_variable_households_data_iteration = 0;
        

	}
}

xmachine_memory_ChurchMembership* h_allocate_agent_ChurchMembership(){
	xmachine_memory_ChurchMembership* agent = (xmachine_memory_ChurchMembership*)malloc(sizeof(xmachine_memory_ChurchMembership));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_ChurchMembership));

	return agent;
}
void h_free_agent_ChurchMembership(xmachine_memory_ChurchMembership** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_ChurchMembership** h_allocate_agent_ChurchMembership_array(unsigned int count){
	xmachine_memory_ChurchMembership ** agents = (xmachine_memory_ChurchMembership**)malloc(count * sizeof(xmachine_memory_ChurchMembership*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_ChurchMembership();
	}
	return agents;
}
void h_free_agent_ChurchMembership_array(xmachine_memory_ChurchMembership*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_ChurchMembership(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_ChurchMembership_AoS_to_SoA(xmachine_memory_ChurchMembership_list * dst, xmachine_memory_ChurchMembership** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->church_id[i] = src[i]->church_id;
			 
			dst->household_id[i] = src[i]->household_id;
			 
			dst->churchdur[i] = src[i]->churchdur;
			
		}
	}
}


void h_add_agent_ChurchMembership_chumembershipdefault(xmachine_memory_ChurchMembership* agent){
	if (h_xmachine_memory_ChurchMembership_count + 1 > xmachine_memory_ChurchMembership_MAX){
		printf("Error: Buffer size of ChurchMembership agents in state chumembershipdefault will be exceeded by h_add_agent_ChurchMembership_chumembershipdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_ChurchMembership_hostToDevice(d_ChurchMemberships_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_ChurchMembership_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_ChurchMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_ChurchMemberships_chumembershipdefault, d_ChurchMemberships_new, h_xmachine_memory_ChurchMembership_chumembershipdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_ChurchMembership_chumembershipdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_ChurchMembership_chumembershipdefault_count, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration = 0;
    

}
void h_add_agents_ChurchMembership_chumembershipdefault(xmachine_memory_ChurchMembership** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_ChurchMembership_count + count > xmachine_memory_ChurchMembership_MAX){
			printf("Error: Buffer size of ChurchMembership agents in state chumembershipdefault will be exceeded by h_add_agents_ChurchMembership_chumembershipdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_ChurchMembership_AoS_to_SoA(h_ChurchMemberships_chumembershipdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_ChurchMembership_hostToDevice(d_ChurchMemberships_new, h_ChurchMemberships_chumembershipdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_ChurchMembership_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_ChurchMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_ChurchMemberships_chumembershipdefault, d_ChurchMemberships_new, h_xmachine_memory_ChurchMembership_chumembershipdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_ChurchMembership_chumembershipdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_ChurchMembership_chumembershipdefault_count, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration = 0;
        h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration = 0;
        h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration = 0;
        

	}
}

xmachine_memory_Transport* h_allocate_agent_Transport(){
	xmachine_memory_Transport* agent = (xmachine_memory_Transport*)malloc(sizeof(xmachine_memory_Transport));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Transport));

	return agent;
}
void h_free_agent_Transport(xmachine_memory_Transport** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Transport** h_allocate_agent_Transport_array(unsigned int count){
	xmachine_memory_Transport ** agents = (xmachine_memory_Transport**)malloc(count * sizeof(xmachine_memory_Transport*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Transport();
	}
	return agents;
}
void h_free_agent_Transport_array(xmachine_memory_Transport*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Transport(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Transport_AoS_to_SoA(xmachine_memory_Transport_list * dst, xmachine_memory_Transport** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->step[i] = src[i]->step;
			 
			dst->duration[i] = src[i]->duration;
			
		}
	}
}


void h_add_agent_Transport_trdefault(xmachine_memory_Transport* agent){
	if (h_xmachine_memory_Transport_count + 1 > xmachine_memory_Transport_MAX){
		printf("Error: Buffer size of Transport agents in state trdefault will be exceeded by h_add_agent_Transport_trdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Transport_hostToDevice(d_Transports_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Transport_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Transport_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Transports_trdefault, d_Transports_new, h_xmachine_memory_Transport_trdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Transport_trdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Transports_trdefault_variable_id_data_iteration = 0;
    h_Transports_trdefault_variable_step_data_iteration = 0;
    h_Transports_trdefault_variable_duration_data_iteration = 0;
    

}
void h_add_agents_Transport_trdefault(xmachine_memory_Transport** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Transport_count + count > xmachine_memory_Transport_MAX){
			printf("Error: Buffer size of Transport agents in state trdefault will be exceeded by h_add_agents_Transport_trdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Transport_AoS_to_SoA(h_Transports_trdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Transport_hostToDevice(d_Transports_new, h_Transports_trdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Transport_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Transport_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Transports_trdefault, d_Transports_new, h_xmachine_memory_Transport_trdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Transport_trdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Transports_trdefault_variable_id_data_iteration = 0;
        h_Transports_trdefault_variable_step_data_iteration = 0;
        h_Transports_trdefault_variable_duration_data_iteration = 0;
        

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
unsigned int reduce_Person_default_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->step),  thrust::device_pointer_cast(d_Persons_default->step) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_step_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->step),  thrust::device_pointer_cast(d_Persons_default->step) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_step_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_step_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->step);
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
unsigned int reduce_Person_default_churchfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->churchfreq),  thrust::device_pointer_cast(d_Persons_default->churchfreq) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_churchfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->churchfreq),  thrust::device_pointer_cast(d_Persons_default->churchfreq) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_churchfreq_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_churchfreq_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_default_churchdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->churchdur),  thrust::device_pointer_cast(d_Persons_default->churchdur) + h_xmachine_memory_Person_default_count);
}

float min_Person_default_churchdur_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_default_churchdur_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_transportuser_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportuser),  thrust::device_pointer_cast(d_Persons_default->transportuser) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_transportuser_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportuser),  thrust::device_pointer_cast(d_Persons_default->transportuser) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_transportuser_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportuser);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_transportuser_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportuser);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_transportfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportfreq),  thrust::device_pointer_cast(d_Persons_default->transportfreq) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_transportfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportfreq),  thrust::device_pointer_cast(d_Persons_default->transportfreq) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_transportfreq_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_transportfreq_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_transportdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportdur),  thrust::device_pointer_cast(d_Persons_default->transportdur) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_transportdur_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportdur),  thrust::device_pointer_cast(d_Persons_default->transportdur) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_transportdur_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_transportdur_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_household_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->household),  thrust::device_pointer_cast(d_Persons_default->household) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_household_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->household),  thrust::device_pointer_cast(d_Persons_default->household) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_household_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->household);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_household_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->household);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_church_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->church),  thrust::device_pointer_cast(d_Persons_default->church) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_church_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->church),  thrust::device_pointer_cast(d_Persons_default->church) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_church_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->church);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_church_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->church);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_busy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->busy),  thrust::device_pointer_cast(d_Persons_default->busy) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_busy_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->busy),  thrust::device_pointer_cast(d_Persons_default->busy) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_busy_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->busy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_busy_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->busy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_startstep_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->startstep),  thrust::device_pointer_cast(d_Persons_default->startstep) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_startstep_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->startstep),  thrust::device_pointer_cast(d_Persons_default->startstep) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_startstep_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->startstep);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_startstep_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->startstep);
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
unsigned int reduce_Person_s2_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->step),  thrust::device_pointer_cast(d_Persons_s2->step) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_step_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->step),  thrust::device_pointer_cast(d_Persons_s2->step) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_step_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_step_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->step);
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
unsigned int reduce_Person_s2_churchfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->churchfreq),  thrust::device_pointer_cast(d_Persons_s2->churchfreq) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_churchfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->churchfreq),  thrust::device_pointer_cast(d_Persons_s2->churchfreq) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_churchfreq_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_churchfreq_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_s2_churchdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->churchdur),  thrust::device_pointer_cast(d_Persons_s2->churchdur) + h_xmachine_memory_Person_s2_count);
}

float min_Person_s2_churchdur_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_s2_churchdur_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_transportuser_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportuser),  thrust::device_pointer_cast(d_Persons_s2->transportuser) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_transportuser_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportuser),  thrust::device_pointer_cast(d_Persons_s2->transportuser) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_transportuser_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportuser);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_transportuser_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportuser);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_transportfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportfreq),  thrust::device_pointer_cast(d_Persons_s2->transportfreq) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_transportfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportfreq),  thrust::device_pointer_cast(d_Persons_s2->transportfreq) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_transportfreq_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_transportfreq_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_transportdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportdur),  thrust::device_pointer_cast(d_Persons_s2->transportdur) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_transportdur_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportdur),  thrust::device_pointer_cast(d_Persons_s2->transportdur) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_transportdur_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_transportdur_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_household_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->household),  thrust::device_pointer_cast(d_Persons_s2->household) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_household_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->household),  thrust::device_pointer_cast(d_Persons_s2->household) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_household_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->household);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_household_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->household);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_church_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->church),  thrust::device_pointer_cast(d_Persons_s2->church) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_church_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->church),  thrust::device_pointer_cast(d_Persons_s2->church) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_church_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->church);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_church_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->church);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_busy_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->busy),  thrust::device_pointer_cast(d_Persons_s2->busy) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_busy_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->busy),  thrust::device_pointer_cast(d_Persons_s2->busy) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_busy_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->busy);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_busy_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->busy);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_startstep_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->startstep),  thrust::device_pointer_cast(d_Persons_s2->startstep) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_startstep_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->startstep),  thrust::device_pointer_cast(d_Persons_s2->startstep) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_startstep_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->startstep);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_startstep_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->startstep);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->id),  thrust::device_pointer_cast(d_Households_hhdefault->id) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->id),  thrust::device_pointer_cast(d_Households_hhdefault->id) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->step),  thrust::device_pointer_cast(d_Households_hhdefault->step) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_step_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->step),  thrust::device_pointer_cast(d_Households_hhdefault->step) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_step_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_step_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->step);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->size),  thrust::device_pointer_cast(d_Households_hhdefault->size) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_size_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->size),  thrust::device_pointer_cast(d_Households_hhdefault->size) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->size);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_churchgoing_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->churchgoing),  thrust::device_pointer_cast(d_Households_hhdefault->churchgoing) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_churchgoing_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->churchgoing),  thrust::device_pointer_cast(d_Households_hhdefault->churchgoing) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_churchgoing_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->churchgoing);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_churchgoing_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->churchgoing);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_churchfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->churchfreq),  thrust::device_pointer_cast(d_Households_hhdefault->churchfreq) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_churchfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->churchfreq),  thrust::device_pointer_cast(d_Households_hhdefault->churchfreq) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_churchfreq_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->churchfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_churchfreq_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->churchfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_adults_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->adults),  thrust::device_pointer_cast(d_Households_hhdefault->adults) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_adults_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->adults),  thrust::device_pointer_cast(d_Households_hhdefault->adults) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_adults_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->adults);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_adults_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->adults);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count);
}

unsigned int count_HouseholdMembership_hhmembershipdefault_household_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count_value);
}
unsigned int min_HouseholdMembership_hhmembershipdefault_household_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_HouseholdMembership_hhmembershipdefault_household_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_HouseholdMembership_hhmembershipdefault_person_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count);
}

unsigned int count_HouseholdMembership_hhmembershipdefault_person_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count_value);
}
unsigned int min_HouseholdMembership_hhmembershipdefault_person_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_HouseholdMembership_hhmembershipdefault_person_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->person_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchgoing_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count);
}

unsigned int count_HouseholdMembership_hhmembershipdefault_churchgoing_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count_value);
}
unsigned int min_HouseholdMembership_hhmembershipdefault_churchgoing_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_HouseholdMembership_hhmembershipdefault_churchgoing_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchgoing);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchfreq_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count);
}

unsigned int count_HouseholdMembership_hhmembershipdefault_churchfreq_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count_value);
}
unsigned int min_HouseholdMembership_hhmembershipdefault_churchfreq_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_HouseholdMembership_hhmembershipdefault_churchfreq_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->churchfreq);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Church_chudefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->id),  thrust::device_pointer_cast(d_Churchs_chudefault->id) + h_xmachine_memory_Church_chudefault_count);
}

unsigned int count_Church_chudefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Churchs_chudefault->id),  thrust::device_pointer_cast(d_Churchs_chudefault->id) + h_xmachine_memory_Church_chudefault_count, count_value);
}
unsigned int min_Church_chudefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Church_chudefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Church_chudefault_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->step),  thrust::device_pointer_cast(d_Churchs_chudefault->step) + h_xmachine_memory_Church_chudefault_count);
}

unsigned int count_Church_chudefault_step_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Churchs_chudefault->step),  thrust::device_pointer_cast(d_Churchs_chudefault->step) + h_xmachine_memory_Church_chudefault_count, count_value);
}
unsigned int min_Church_chudefault_step_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Church_chudefault_step_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->step);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Church_chudefault_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->size),  thrust::device_pointer_cast(d_Churchs_chudefault->size) + h_xmachine_memory_Church_chudefault_count);
}

unsigned int count_Church_chudefault_size_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Churchs_chudefault->size),  thrust::device_pointer_cast(d_Churchs_chudefault->size) + h_xmachine_memory_Church_chudefault_count, count_value);
}
unsigned int min_Church_chudefault_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Church_chudefault_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->size);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Church_chudefault_duration_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->duration),  thrust::device_pointer_cast(d_Churchs_chudefault->duration) + h_xmachine_memory_Church_chudefault_count);
}

float min_Church_chudefault_duration_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->duration);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Church_chudefault_duration_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->duration);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_ChurchMembership_chumembershipdefault_church_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id),  thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count);
}

unsigned int count_ChurchMembership_chumembershipdefault_church_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id),  thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count, count_value);
}
unsigned int min_ChurchMembership_chumembershipdefault_church_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_ChurchMembership_chumembershipdefault_church_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->church_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_ChurchMembership_chumembershipdefault_household_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id),  thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count);
}

unsigned int count_ChurchMembership_chumembershipdefault_household_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id),  thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count, count_value);
}
unsigned int min_ChurchMembership_chumembershipdefault_household_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_ChurchMembership_chumembershipdefault_household_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->household_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_ChurchMembership_chumembershipdefault_churchdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->churchdur),  thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->churchdur) + h_xmachine_memory_ChurchMembership_chumembershipdefault_count);
}

float min_ChurchMembership_chumembershipdefault_churchdur_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->churchdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_ChurchMembership_chumembershipdefault_churchdur_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_ChurchMemberships_chumembershipdefault->churchdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_ChurchMembership_chumembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Transport_trdefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Transports_trdefault->id),  thrust::device_pointer_cast(d_Transports_trdefault->id) + h_xmachine_memory_Transport_trdefault_count);
}

unsigned int count_Transport_trdefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Transports_trdefault->id),  thrust::device_pointer_cast(d_Transports_trdefault->id) + h_xmachine_memory_Transport_trdefault_count, count_value);
}
unsigned int min_Transport_trdefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Transport_trdefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Transport_trdefault_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Transports_trdefault->step),  thrust::device_pointer_cast(d_Transports_trdefault->step) + h_xmachine_memory_Transport_trdefault_count);
}

unsigned int count_Transport_trdefault_step_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Transports_trdefault->step),  thrust::device_pointer_cast(d_Transports_trdefault->step) + h_xmachine_memory_Transport_trdefault_count, count_value);
}
unsigned int min_Transport_trdefault_step_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Transport_trdefault_step_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->step);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Transport_trdefault_duration_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Transports_trdefault->duration),  thrust::device_pointer_cast(d_Transports_trdefault->duration) + h_xmachine_memory_Transport_trdefault_count);
}

unsigned int count_Transport_trdefault_duration_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Transports_trdefault->duration),  thrust::device_pointer_cast(d_Transports_trdefault->duration) + h_xmachine_memory_Transport_trdefault_count, count_value);
}
unsigned int min_Transport_trdefault_duration_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->duration);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Transport_trdefault_duration_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->duration);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
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
	
	if (h_xmachine_memory_Person_s2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Person_s2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Person_list* Persons_s2_temp = d_Persons;
	d_Persons = d_Persons_s2;
	d_Persons_s2 = Persons_s2_temp;
	//set working count to current state count
	h_xmachine_memory_Person_count = h_xmachine_memory_Person_s2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_count, &h_xmachine_memory_Person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Person_s2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
	
 

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
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of update agents in state s2 will be exceeded moving working agents to next state in function update\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Person_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_s2, d_Persons, h_xmachine_memory_Person_s2_count, h_xmachine_memory_Person_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Person_s2_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Person_init_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_household_membership));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_init
 * Agent function prototype for init function of Person agent
 */
void Person_init(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_init, Person_init_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_init_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (init)
	//Reallocate   : false
	//Input        : household_membership
	//Output       : 
	//Agent Output : 
	GPUFLAME_init<<<g, b, sm_size, stream>>>(d_Persons, d_household_memberships);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of init agents in state s2 will be exceeded moving working agents to next state in function init\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Person_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_s2, d_Persons, h_xmachine_memory_Person_s2_count, h_xmachine_memory_Person_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Person_s2_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Household_hhupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Household_hhupdate
 * Agent function prototype for hhupdate function of Household agent
 */
void Household_hhupdate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Household_hhdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Household_hhdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Household_list* Households_hhdefault_temp = d_Households;
	d_Households = d_Households_hhdefault;
	d_Households_hhdefault = Households_hhdefault_temp;
	//set working count to current state count
	h_xmachine_memory_Household_count = h_xmachine_memory_Household_hhdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_count, &h_xmachine_memory_Household_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Household_hhdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_hhdefault_count, &h_xmachine_memory_Household_hhdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_hhupdate, Household_hhupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Household_hhupdate_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (hhupdate)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_hhupdate<<<g, b, sm_size, stream>>>(d_Households);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Household_hhdefault_count+h_xmachine_memory_Household_count > xmachine_memory_Household_MAX){
		printf("Error: Buffer size of hhupdate agents in state hhdefault will be exceeded moving working agents to next state in function hhupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Households_hhdefault_temp = d_Households;
  d_Households = d_Households_hhdefault;
  d_Households_hhdefault = Households_hhdefault_temp;
        
	//update new state agent size
	h_xmachine_memory_Household_hhdefault_count += h_xmachine_memory_Household_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_hhdefault_count, &h_xmachine_memory_Household_hhdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int HouseholdMembership_hhinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_church_membership));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** HouseholdMembership_hhinit
 * Agent function prototype for hhinit function of HouseholdMembership agent
 */
void HouseholdMembership_hhinit(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_HouseholdMembership_list* HouseholdMemberships_hhmembershipdefault_temp = d_HouseholdMemberships;
	d_HouseholdMemberships = d_HouseholdMemberships_hhmembershipdefault;
	d_HouseholdMemberships_hhmembershipdefault = HouseholdMemberships_hhmembershipdefault_temp;
	//set working count to current state count
	h_xmachine_memory_HouseholdMembership_count = h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_HouseholdMembership_count, &h_xmachine_memory_HouseholdMembership_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_household_membership_count + h_xmachine_memory_HouseholdMembership_count > xmachine_message_household_membership_MAX){
		printf("Error: Buffer size of household_membership message will be exceeded in function hhinit\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_hhinit, HouseholdMembership_hhinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = HouseholdMembership_hhinit_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_household_membership_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_household_membership_output_type, &h_message_household_membership_output_type, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_HouseholdMembership_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_HouseholdMembership_scan_input<<<gridSize, blockSize, 0, stream>>>(d_HouseholdMemberships);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (hhinit)
	//Reallocate   : true
	//Input        : church_membership
	//Output       : household_membership
	//Agent Output : 
	GPUFLAME_hhinit<<<g, b, sm_size, stream>>>(d_HouseholdMemberships, d_church_memberships, d_household_memberships);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_household_membership_count += h_xmachine_memory_HouseholdMembership_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_household_membership_count, &h_message_household_membership_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_HouseholdMembership, 
        temp_scan_storage_bytes_HouseholdMembership, 
        d_HouseholdMemberships->_scan_input,
        d_HouseholdMemberships->_position,
        h_xmachine_memory_HouseholdMembership_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_HouseholdMembership_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_HouseholdMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_HouseholdMemberships_swap, d_HouseholdMemberships, 0, h_xmachine_memory_HouseholdMembership_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_HouseholdMembership_list* hhinit_HouseholdMemberships_temp = d_HouseholdMemberships;
	d_HouseholdMemberships = d_HouseholdMemberships_swap;
	d_HouseholdMemberships_swap = hhinit_HouseholdMemberships_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_HouseholdMemberships_swap->_position[h_xmachine_memory_HouseholdMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_HouseholdMemberships_swap->_scan_input[h_xmachine_memory_HouseholdMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_HouseholdMembership_count = scan_last_sum+1;
	else
		h_xmachine_memory_HouseholdMembership_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_HouseholdMembership_count, &h_xmachine_memory_HouseholdMembership_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count+h_xmachine_memory_HouseholdMembership_count > xmachine_memory_HouseholdMembership_MAX){
		printf("Error: Buffer size of hhinit agents in state hhmembershipdefault will be exceeded moving working agents to next state in function hhinit\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_HouseholdMembership_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_HouseholdMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_HouseholdMemberships_hhmembershipdefault, d_HouseholdMemberships, h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, h_xmachine_memory_HouseholdMembership_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count += h_xmachine_memory_HouseholdMembership_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Church_chuupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Church_chuupdate
 * Agent function prototype for chuupdate function of Church agent
 */
void Church_chuupdate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Church_chudefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Church_chudefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Church_list* Churchs_chudefault_temp = d_Churchs;
	d_Churchs = d_Churchs_chudefault;
	d_Churchs_chudefault = Churchs_chudefault_temp;
	//set working count to current state count
	h_xmachine_memory_Church_count = h_xmachine_memory_Church_chudefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_count, &h_xmachine_memory_Church_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Church_chudefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_chudefault_count, &h_xmachine_memory_Church_chudefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_chuupdate, Church_chuupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Church_chuupdate_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (chuupdate)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_chuupdate<<<g, b, sm_size, stream>>>(d_Churchs);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Church_chudefault_count+h_xmachine_memory_Church_count > xmachine_memory_Church_MAX){
		printf("Error: Buffer size of chuupdate agents in state chudefault will be exceeded moving working agents to next state in function chuupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Churchs_chudefault_temp = d_Churchs;
  d_Churchs = d_Churchs_chudefault;
  d_Churchs_chudefault = Churchs_chudefault_temp;
        
	//update new state agent size
	h_xmachine_memory_Church_chudefault_count += h_xmachine_memory_Church_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_chudefault_count, &h_xmachine_memory_Church_chudefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int ChurchMembership_chuinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** ChurchMembership_chuinit
 * Agent function prototype for chuinit function of ChurchMembership agent
 */
void ChurchMembership_chuinit(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_ChurchMembership_chumembershipdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_ChurchMembership_chumembershipdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_ChurchMembership_list* ChurchMemberships_chumembershipdefault_temp = d_ChurchMemberships;
	d_ChurchMemberships = d_ChurchMemberships_chumembershipdefault;
	d_ChurchMemberships_chumembershipdefault = ChurchMemberships_chumembershipdefault_temp;
	//set working count to current state count
	h_xmachine_memory_ChurchMembership_count = h_xmachine_memory_ChurchMembership_chumembershipdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_ChurchMembership_count, &h_xmachine_memory_ChurchMembership_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_ChurchMembership_chumembershipdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_ChurchMembership_chumembershipdefault_count, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_church_membership_count + h_xmachine_memory_ChurchMembership_count > xmachine_message_church_membership_MAX){
		printf("Error: Buffer size of church_membership message will be exceeded in function chuinit\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_chuinit, ChurchMembership_chuinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = ChurchMembership_chuinit_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_church_membership_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_church_membership_output_type, &h_message_church_membership_output_type, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_ChurchMembership_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_ChurchMembership_scan_input<<<gridSize, blockSize, 0, stream>>>(d_ChurchMemberships);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (chuinit)
	//Reallocate   : true
	//Input        : 
	//Output       : church_membership
	//Agent Output : 
	GPUFLAME_chuinit<<<g, b, sm_size, stream>>>(d_ChurchMemberships, d_church_memberships);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_church_membership_count += h_xmachine_memory_ChurchMembership_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_church_membership_count, &h_message_church_membership_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_ChurchMembership, 
        temp_scan_storage_bytes_ChurchMembership, 
        d_ChurchMemberships->_scan_input,
        d_ChurchMemberships->_position,
        h_xmachine_memory_ChurchMembership_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_ChurchMembership_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_ChurchMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_ChurchMemberships_swap, d_ChurchMemberships, 0, h_xmachine_memory_ChurchMembership_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_ChurchMembership_list* chuinit_ChurchMemberships_temp = d_ChurchMemberships;
	d_ChurchMemberships = d_ChurchMemberships_swap;
	d_ChurchMemberships_swap = chuinit_ChurchMemberships_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_ChurchMemberships_swap->_position[h_xmachine_memory_ChurchMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_ChurchMemberships_swap->_scan_input[h_xmachine_memory_ChurchMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_ChurchMembership_count = scan_last_sum+1;
	else
		h_xmachine_memory_ChurchMembership_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_ChurchMembership_count, &h_xmachine_memory_ChurchMembership_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_ChurchMembership_chumembershipdefault_count+h_xmachine_memory_ChurchMembership_count > xmachine_memory_ChurchMembership_MAX){
		printf("Error: Buffer size of chuinit agents in state chumembershipdefault will be exceeded moving working agents to next state in function chuinit\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_ChurchMembership_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_ChurchMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_ChurchMemberships_chumembershipdefault, d_ChurchMemberships, h_xmachine_memory_ChurchMembership_chumembershipdefault_count, h_xmachine_memory_ChurchMembership_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_ChurchMembership_chumembershipdefault_count += h_xmachine_memory_ChurchMembership_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_ChurchMembership_chumembershipdefault_count, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Transport_trupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Transport_trupdate
 * Agent function prototype for trupdate function of Transport agent
 */
void Transport_trupdate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Transport_trdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Transport_trdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Transport_list* Transports_trdefault_temp = d_Transports;
	d_Transports = d_Transports_trdefault;
	d_Transports_trdefault = Transports_trdefault_temp;
	//set working count to current state count
	h_xmachine_memory_Transport_count = h_xmachine_memory_Transport_trdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_count, &h_xmachine_memory_Transport_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Transport_trdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_trupdate, Transport_trupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Transport_trupdate_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (trupdate)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_trupdate<<<g, b, sm_size, stream>>>(d_Transports);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Transport_trdefault_count+h_xmachine_memory_Transport_count > xmachine_memory_Transport_MAX){
		printf("Error: Buffer size of trupdate agents in state trdefault will be exceeded moving working agents to next state in function trupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Transports_trdefault_temp = d_Transports;
  d_Transports = d_Transports_trdefault;
  d_Transports_trdefault = Transports_trdefault_temp;
        
	//update new state agent size
	h_xmachine_memory_Transport_trdefault_count += h_xmachine_memory_Transport_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));	
	
	
}


 
extern void reset_Person_default_count()
{
    h_xmachine_memory_Person_default_count = 0;
}
 
extern void reset_Person_s2_count()
{
    h_xmachine_memory_Person_s2_count = 0;
}
 
extern void reset_Household_hhdefault_count()
{
    h_xmachine_memory_Household_hhdefault_count = 0;
}
 
extern void reset_HouseholdMembership_hhmembershipdefault_count()
{
    h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count = 0;
}
 
extern void reset_Church_chudefault_count()
{
    h_xmachine_memory_Church_chudefault_count = 0;
}
 
extern void reset_ChurchMembership_chumembershipdefault_count()
{
    h_xmachine_memory_ChurchMembership_chumembershipdefault_count = 0;
}
 
extern void reset_Transport_trdefault_count()
{
    h_xmachine_memory_Transport_trdefault_count = 0;
}
