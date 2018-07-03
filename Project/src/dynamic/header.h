
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


#ifndef __HEADER
#define __HEADER
#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 32768

//Maximum population size of xmachine_memory_Person
#define xmachine_memory_Person_MAX 32768

//Maximum population size of xmachine_memory_Household
#define xmachine_memory_Household_MAX 8192

//Maximum population size of xmachine_memory_Church
#define xmachine_memory_Church_MAX 256 
//Agent variable array length for xmachine_memory_Household->people
#define xmachine_memory_Household_people_LENGTH 32 
//Agent variable array length for xmachine_memory_Church->households
#define xmachine_memory_Church_households_LENGTH 128


  
  
/* Message population size definitions */

/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */


/* Spatial partitioning grid size definitions */
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_Person
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Person
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int age;    /**< X-machine memory variable age of type unsigned int.*/
    unsigned int gender;    /**< X-machine memory variable gender of type unsigned int.*/
    unsigned int householdsize;    /**< X-machine memory variable householdsize of type unsigned int.*/
};

/** struct xmachine_memory_Household
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Household
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    int *people;    /**< X-machine memory variable people of type int.*/
    unsigned int churchgoing;    /**< X-machine memory variable churchgoing of type unsigned int.*/
    unsigned int churchfreq;    /**< X-machine memory variable churchfreq of type unsigned int.*/
    unsigned int adults;    /**< X-machine memory variable adults of type unsigned int.*/
};

/** struct xmachine_memory_Church
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Church
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    float duration;    /**< X-machine memory variable duration of type float.*/
    int *households;    /**< X-machine memory variable households of type int.*/
};



/* Message structures */



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Person_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Person_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Person_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Person_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Person_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int age [xmachine_memory_Person_MAX];    /**< X-machine memory variable list age of type unsigned int.*/
    unsigned int gender [xmachine_memory_Person_MAX];    /**< X-machine memory variable list gender of type unsigned int.*/
    unsigned int householdsize [xmachine_memory_Person_MAX];    /**< X-machine memory variable list householdsize of type unsigned int.*/
};

/** struct xmachine_memory_Household_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Household_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Household_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Household_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Household_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int size [xmachine_memory_Household_MAX];    /**< X-machine memory variable list size of type unsigned int.*/
    int people [xmachine_memory_Household_MAX*32];    /**< X-machine memory variable list people of type int.*/
    unsigned int churchgoing [xmachine_memory_Household_MAX];    /**< X-machine memory variable list churchgoing of type unsigned int.*/
    unsigned int churchfreq [xmachine_memory_Household_MAX];    /**< X-machine memory variable list churchfreq of type unsigned int.*/
    unsigned int adults [xmachine_memory_Household_MAX];    /**< X-machine memory variable list adults of type unsigned int.*/
};

/** struct xmachine_memory_Church_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Church_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Church_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Church_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Church_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int size [xmachine_memory_Church_MAX];    /**< X-machine memory variable list size of type unsigned int.*/
    float duration [xmachine_memory_Church_MAX];    /**< X-machine memory variable list duration of type float.*/
    int households [xmachine_memory_Church_MAX*128];    /**< X-machine memory variable list households of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */



/* Spatially Partitioned Message boundary Matrices */



  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * update FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int update(xmachine_memory_Person* agent, RNG_rand48* rand48);

/**
 * hhupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Household. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int hhupdate(xmachine_memory_Household* agent);

/**
 * chuupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Church. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int chuupdate(xmachine_memory_Church* agent);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Person_agent
 * Adds a new continuous valued Person agent to the xmachine_memory_Person_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Person_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param age	agent agent variable of type unsigned int
 * @param gender	agent agent variable of type unsigned int
 * @param householdsize	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int age, unsigned int gender, unsigned int householdsize);

/** add_Household_agent
 * Adds a new continuous valued Household agent to the xmachine_memory_Household_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Household_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param churchgoing	agent agent variable of type unsigned int
 * @param churchfreq	agent agent variable of type unsigned int
 * @param adults	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, unsigned int size, unsigned int churchgoing, unsigned int churchfreq, unsigned int adults);

/** get_Household_agent_array_value
 *  Template function for accessing Household agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Household_agent_array_value(T *array, unsigned int index);

/** set_Household_agent_array_value
 *  Template function for setting Household agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Household_agent_array_value(T *array, unsigned int index, T value);


  

/** add_Church_agent
 * Adds a new continuous valued Church agent to the xmachine_memory_Church_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Church_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param duration	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float duration);

/** get_Church_agent_array_value
 *  Template function for accessing Church agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Church_agent_array_value(T *array, unsigned int index);

/** set_Church_agent_array_value
 *  Template function for setting Church agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Church_agent_array_value(T *array, unsigned int index, T value);


  


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_Persons Pointer to agent list on the host
 * @param d_Persons Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Person_count Pointer to agent counter
 * @param h_Households Pointer to agent list on the host
 * @param d_Households Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Household_count Pointer to agent counter
 * @param h_Churchs Pointer to agent list on the host
 * @param d_Churchs Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Church_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count,xmachine_memory_Household_list* h_Households_hhdefault, xmachine_memory_Household_list* d_Households_hhdefault, int h_xmachine_memory_Household_hhdefault_count,xmachine_memory_Church_list* h_Churchs_chudefault, xmachine_memory_Church_list* d_Churchs_chudefault, int h_xmachine_memory_Church_chudefault_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Persons Pointer to agent list on the host
 * @param h_xmachine_memory_Person_count Pointer to agent counter
 * @param h_Households Pointer to agent list on the host
 * @param h_xmachine_memory_Household_count Pointer to agent counter
 * @param h_Churchs Pointer to agent list on the host
 * @param h_xmachine_memory_Church_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count,xmachine_memory_Household_list* h_Households, int* h_xmachine_memory_Household_count,xmachine_memory_Church_list* h_Churchs, int* h_xmachine_memory_Church_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Person_MAX_count
 * Gets the max agent count for the Person agent type 
 * @return		the maximum Person agent count
 */
extern int get_agent_Person_MAX_count();



/** get_agent_Person_default_count
 * Gets the agent count for the Person agent type in state default
 * @return		the current Person agent count in state default
 */
extern int get_agent_Person_default_count();

/** reset_default_count
 * Resets the agent count of the Person in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Person_default_count();

/** get_device_Person_default_agents
 * Gets a pointer to xmachine_memory_Person_list on the GPU device
 * @return		a xmachine_memory_Person_list on the GPU device
 */
extern xmachine_memory_Person_list* get_device_Person_default_agents();

/** get_host_Person_default_agents
 * Gets a pointer to xmachine_memory_Person_list on the CPU host
 * @return		a xmachine_memory_Person_list on the CPU host
 */
extern xmachine_memory_Person_list* get_host_Person_default_agents();


/** sort_Persons_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Persons_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Person_list* agents));


/** get_agent_Person_s2_count
 * Gets the agent count for the Person agent type in state s2
 * @return		the current Person agent count in state s2
 */
extern int get_agent_Person_s2_count();

/** reset_s2_count
 * Resets the agent count of the Person in state s2 to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Person_s2_count();

/** get_device_Person_s2_agents
 * Gets a pointer to xmachine_memory_Person_list on the GPU device
 * @return		a xmachine_memory_Person_list on the GPU device
 */
extern xmachine_memory_Person_list* get_device_Person_s2_agents();

/** get_host_Person_s2_agents
 * Gets a pointer to xmachine_memory_Person_list on the CPU host
 * @return		a xmachine_memory_Person_list on the CPU host
 */
extern xmachine_memory_Person_list* get_host_Person_s2_agents();


/** sort_Persons_s2
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Persons_s2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Person_list* agents));


    
/** get_agent_Household_MAX_count
 * Gets the max agent count for the Household agent type 
 * @return		the maximum Household agent count
 */
extern int get_agent_Household_MAX_count();



/** get_agent_Household_hhdefault_count
 * Gets the agent count for the Household agent type in state hhdefault
 * @return		the current Household agent count in state hhdefault
 */
extern int get_agent_Household_hhdefault_count();

/** reset_hhdefault_count
 * Resets the agent count of the Household in state hhdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Household_hhdefault_count();

/** get_device_Household_hhdefault_agents
 * Gets a pointer to xmachine_memory_Household_list on the GPU device
 * @return		a xmachine_memory_Household_list on the GPU device
 */
extern xmachine_memory_Household_list* get_device_Household_hhdefault_agents();

/** get_host_Household_hhdefault_agents
 * Gets a pointer to xmachine_memory_Household_list on the CPU host
 * @return		a xmachine_memory_Household_list on the CPU host
 */
extern xmachine_memory_Household_list* get_host_Household_hhdefault_agents();


/** sort_Households_hhdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Households_hhdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Household_list* agents));


    
/** get_agent_Church_MAX_count
 * Gets the max agent count for the Church agent type 
 * @return		the maximum Church agent count
 */
extern int get_agent_Church_MAX_count();



/** get_agent_Church_chudefault_count
 * Gets the agent count for the Church agent type in state chudefault
 * @return		the current Church agent count in state chudefault
 */
extern int get_agent_Church_chudefault_count();

/** reset_chudefault_count
 * Resets the agent count of the Church in state chudefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Church_chudefault_count();

/** get_device_Church_chudefault_agents
 * Gets a pointer to xmachine_memory_Church_list on the GPU device
 * @return		a xmachine_memory_Church_list on the GPU device
 */
extern xmachine_memory_Church_list* get_device_Church_chudefault_agents();

/** get_host_Church_chudefault_agents
 * Gets a pointer to xmachine_memory_Church_list on the CPU host
 * @return		a xmachine_memory_Church_list on the CPU host
 */
extern xmachine_memory_Church_list* get_host_Church_chudefault_agents();


/** sort_Churchs_chudefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Churchs_chudefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Church_list* agents));



/* Host based access of agent variables*/

/** unsigned int get_Person_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_default_variable_id(unsigned int index);

/** unsigned int get_Person_default_variable_age(unsigned int index)
 * Gets the value of the age variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Person_default_variable_age(unsigned int index);

/** unsigned int get_Person_default_variable_gender(unsigned int index)
 * Gets the value of the gender variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable gender
 */
__host__ unsigned int get_Person_default_variable_gender(unsigned int index);

/** unsigned int get_Person_default_variable_householdsize(unsigned int index)
 * Gets the value of the householdsize variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdsize
 */
__host__ unsigned int get_Person_default_variable_householdsize(unsigned int index);

/** unsigned int get_Person_s2_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_s2_variable_id(unsigned int index);

/** unsigned int get_Person_s2_variable_age(unsigned int index)
 * Gets the value of the age variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Person_s2_variable_age(unsigned int index);

/** unsigned int get_Person_s2_variable_gender(unsigned int index)
 * Gets the value of the gender variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable gender
 */
__host__ unsigned int get_Person_s2_variable_gender(unsigned int index);

/** unsigned int get_Person_s2_variable_householdsize(unsigned int index)
 * Gets the value of the householdsize variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdsize
 */
__host__ unsigned int get_Person_s2_variable_householdsize(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Household_hhdefault_variable_id(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_size(unsigned int index)
 * Gets the value of the size variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_Household_hhdefault_variable_size(unsigned int index);

/** int get_Household_hhdefault_variable_people(unsigned int index, unsigned int element)
 * Gets the element-th value of the people variable array of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable people
 */
__host__ int get_Household_hhdefault_variable_people(unsigned int index, unsigned int element);

/** unsigned int get_Household_hhdefault_variable_churchgoing(unsigned int index)
 * Gets the value of the churchgoing variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchgoing
 */
__host__ unsigned int get_Household_hhdefault_variable_churchgoing(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Household_hhdefault_variable_churchfreq(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_adults(unsigned int index)
 * Gets the value of the adults variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable adults
 */
__host__ unsigned int get_Household_hhdefault_variable_adults(unsigned int index);

/** unsigned int get_Church_chudefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Church_chudefault_variable_id(unsigned int index);

/** unsigned int get_Church_chudefault_variable_size(unsigned int index)
 * Gets the value of the size variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_Church_chudefault_variable_size(unsigned int index);

/** float get_Church_chudefault_variable_duration(unsigned int index)
 * Gets the value of the duration variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable duration
 */
__host__ float get_Church_chudefault_variable_duration(unsigned int index);

/** int get_Church_chudefault_variable_households(unsigned int index, unsigned int element)
 * Gets the element-th value of the households variable array of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable households
 */
__host__ int get_Church_chudefault_variable_households(unsigned int index, unsigned int element);




/* Host based agent creation functions */

/** h_allocate_agent_Person
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Person struct.
 */
xmachine_memory_Person* h_allocate_agent_Person();
/** h_free_agent_Person
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Person(xmachine_memory_Person** agent);
/** h_allocate_agent_Person_array
 * Utility function to allocate an array of structs for  Person agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Person** h_allocate_agent_Person_array(unsigned int count);
/** h_free_agent_Person_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Person_array(xmachine_memory_Person*** agents, unsigned int count);


/** h_add_agent_Person_default
 * Host function to add a single agent of type Person to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Person_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Person_default(xmachine_memory_Person* agent);

/** h_add_agents_Person_default(
 * Host function to add multiple agents of type Person to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Person agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Person_default(xmachine_memory_Person** agents, unsigned int count);


/** h_add_agent_Person_s2
 * Host function to add a single agent of type Person to the s2 state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Person_s2 instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Person_s2(xmachine_memory_Person* agent);

/** h_add_agents_Person_s2(
 * Host function to add multiple agents of type Person to the s2 state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Person agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Person_s2(xmachine_memory_Person** agents, unsigned int count);

/** h_allocate_agent_Household
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Household struct.
 */
xmachine_memory_Household* h_allocate_agent_Household();
/** h_free_agent_Household
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Household(xmachine_memory_Household** agent);
/** h_allocate_agent_Household_array
 * Utility function to allocate an array of structs for  Household agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Household** h_allocate_agent_Household_array(unsigned int count);
/** h_free_agent_Household_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Household_array(xmachine_memory_Household*** agents, unsigned int count);


/** h_add_agent_Household_hhdefault
 * Host function to add a single agent of type Household to the hhdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Household_hhdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Household_hhdefault(xmachine_memory_Household* agent);

/** h_add_agents_Household_hhdefault(
 * Host function to add multiple agents of type Household to the hhdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Household agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Household_hhdefault(xmachine_memory_Household** agents, unsigned int count);

/** h_allocate_agent_Church
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Church struct.
 */
xmachine_memory_Church* h_allocate_agent_Church();
/** h_free_agent_Church
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Church(xmachine_memory_Church** agent);
/** h_allocate_agent_Church_array
 * Utility function to allocate an array of structs for  Church agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Church** h_allocate_agent_Church_array(unsigned int count);
/** h_free_agent_Church_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Church_array(xmachine_memory_Church*** agents, unsigned int count);


/** h_add_agent_Church_chudefault
 * Host function to add a single agent of type Church to the chudefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Church_chudefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Church_chudefault(xmachine_memory_Church* agent);

/** h_add_agents_Church_chudefault(
 * Host function to add multiple agents of type Church to the chudefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Church agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Church_chudefault(xmachine_memory_Church** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** unsigned int reduce_Person_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_id_variable();



/** unsigned int count_Person_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_id_variable(int count_value);

/** unsigned int min_Person_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_id_variable();
/** unsigned int max_Person_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_id_variable();

/** unsigned int reduce_Person_default_age_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_age_variable();



/** unsigned int count_Person_default_age_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_age_variable(int count_value);

/** unsigned int min_Person_default_age_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_age_variable();
/** unsigned int max_Person_default_age_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_age_variable();

/** unsigned int reduce_Person_default_gender_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_gender_variable();



/** unsigned int count_Person_default_gender_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_gender_variable(int count_value);

/** unsigned int min_Person_default_gender_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_gender_variable();
/** unsigned int max_Person_default_gender_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_gender_variable();

/** unsigned int reduce_Person_default_householdsize_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_householdsize_variable();



/** unsigned int count_Person_default_householdsize_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_householdsize_variable(int count_value);

/** unsigned int min_Person_default_householdsize_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_householdsize_variable();
/** unsigned int max_Person_default_householdsize_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_householdsize_variable();

/** unsigned int reduce_Person_s2_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_id_variable();



/** unsigned int count_Person_s2_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_id_variable(int count_value);

/** unsigned int min_Person_s2_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_id_variable();
/** unsigned int max_Person_s2_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_id_variable();

/** unsigned int reduce_Person_s2_age_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_age_variable();



/** unsigned int count_Person_s2_age_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_age_variable(int count_value);

/** unsigned int min_Person_s2_age_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_age_variable();
/** unsigned int max_Person_s2_age_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_age_variable();

/** unsigned int reduce_Person_s2_gender_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_gender_variable();



/** unsigned int count_Person_s2_gender_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_gender_variable(int count_value);

/** unsigned int min_Person_s2_gender_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_gender_variable();
/** unsigned int max_Person_s2_gender_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_gender_variable();

/** unsigned int reduce_Person_s2_householdsize_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_householdsize_variable();



/** unsigned int count_Person_s2_householdsize_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_householdsize_variable(int count_value);

/** unsigned int min_Person_s2_householdsize_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_householdsize_variable();
/** unsigned int max_Person_s2_householdsize_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_householdsize_variable();

/** unsigned int reduce_Household_hhdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_id_variable();



/** unsigned int count_Household_hhdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_id_variable(int count_value);

/** unsigned int min_Household_hhdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_id_variable();
/** unsigned int max_Household_hhdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_id_variable();

/** unsigned int reduce_Household_hhdefault_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_size_variable();



/** unsigned int count_Household_hhdefault_size_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_size_variable(int count_value);

/** unsigned int min_Household_hhdefault_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_size_variable();
/** unsigned int max_Household_hhdefault_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_size_variable();

/** unsigned int reduce_Household_hhdefault_churchgoing_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_churchgoing_variable();



/** unsigned int count_Household_hhdefault_churchgoing_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_churchgoing_variable(int count_value);

/** unsigned int min_Household_hhdefault_churchgoing_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_churchgoing_variable();
/** unsigned int max_Household_hhdefault_churchgoing_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_churchgoing_variable();

/** unsigned int reduce_Household_hhdefault_churchfreq_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_churchfreq_variable();



/** unsigned int count_Household_hhdefault_churchfreq_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_churchfreq_variable(int count_value);

/** unsigned int min_Household_hhdefault_churchfreq_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_churchfreq_variable();
/** unsigned int max_Household_hhdefault_churchfreq_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_churchfreq_variable();

/** unsigned int reduce_Household_hhdefault_adults_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_adults_variable();



/** unsigned int count_Household_hhdefault_adults_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_adults_variable(int count_value);

/** unsigned int min_Household_hhdefault_adults_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_adults_variable();
/** unsigned int max_Household_hhdefault_adults_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_adults_variable();

/** unsigned int reduce_Church_chudefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Church_chudefault_id_variable();



/** unsigned int count_Church_chudefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Church_chudefault_id_variable(int count_value);

/** unsigned int min_Church_chudefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Church_chudefault_id_variable();
/** unsigned int max_Church_chudefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Church_chudefault_id_variable();

/** unsigned int reduce_Church_chudefault_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Church_chudefault_size_variable();



/** unsigned int count_Church_chudefault_size_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Church_chudefault_size_variable(int count_value);

/** unsigned int min_Church_chudefault_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Church_chudefault_size_variable();
/** unsigned int max_Church_chudefault_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Church_chudefault_size_variable();

/** float reduce_Church_chudefault_duration_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Church_chudefault_duration_variable();



/** float min_Church_chudefault_duration_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Church_chudefault_duration_variable();
/** float max_Church_chudefault_duration_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Church_chudefault_duration_variable();


  
/* global constant variables */

__constant__ float TIME_STEP;

__constant__ float SCALE_FACTOR;

__constant__ unsigned int MAX_AGE;

__constant__ float STARTING_POPULATION;

__constant__ float CHURCH_BETA0;

__constant__ float CHURCH_BETA1;

__constant__ unsigned int CHURCH_K1;

__constant__ unsigned int CHURCH_K2;

__constant__ unsigned int CHURCH_K3;

__constant__ float CHURCH_P1;

__constant__ float CHURCH_P2;

__constant__ float CHURCH_PROB0;

__constant__ float CHURCH_PROB1;

__constant__ float CHURCH_PROB2;

__constant__ float CHURCH_PROB3;

__constant__ float CHURCH_PROB4;

__constant__ float CHURCH_PROB5;

__constant__ float CHURCH_PROB6;

__constant__ float CHURCH_DURATION;

/** set_TIME_STEP
 * Sets the constant variable TIME_STEP on the device which can then be used in the agent functions.
 * @param h_TIME_STEP value to set the variable
 */
extern void set_TIME_STEP(float* h_TIME_STEP);

extern const float* get_TIME_STEP();


extern float h_env_TIME_STEP;

/** set_SCALE_FACTOR
 * Sets the constant variable SCALE_FACTOR on the device which can then be used in the agent functions.
 * @param h_SCALE_FACTOR value to set the variable
 */
extern void set_SCALE_FACTOR(float* h_SCALE_FACTOR);

extern const float* get_SCALE_FACTOR();


extern float h_env_SCALE_FACTOR;

/** set_MAX_AGE
 * Sets the constant variable MAX_AGE on the device which can then be used in the agent functions.
 * @param h_MAX_AGE value to set the variable
 */
extern void set_MAX_AGE(unsigned int* h_MAX_AGE);

extern const unsigned int* get_MAX_AGE();


extern unsigned int h_env_MAX_AGE;

/** set_STARTING_POPULATION
 * Sets the constant variable STARTING_POPULATION on the device which can then be used in the agent functions.
 * @param h_STARTING_POPULATION value to set the variable
 */
extern void set_STARTING_POPULATION(float* h_STARTING_POPULATION);

extern const float* get_STARTING_POPULATION();


extern float h_env_STARTING_POPULATION;

/** set_CHURCH_BETA0
 * Sets the constant variable CHURCH_BETA0 on the device which can then be used in the agent functions.
 * @param h_CHURCH_BETA0 value to set the variable
 */
extern void set_CHURCH_BETA0(float* h_CHURCH_BETA0);

extern const float* get_CHURCH_BETA0();


extern float h_env_CHURCH_BETA0;

/** set_CHURCH_BETA1
 * Sets the constant variable CHURCH_BETA1 on the device which can then be used in the agent functions.
 * @param h_CHURCH_BETA1 value to set the variable
 */
extern void set_CHURCH_BETA1(float* h_CHURCH_BETA1);

extern const float* get_CHURCH_BETA1();


extern float h_env_CHURCH_BETA1;

/** set_CHURCH_K1
 * Sets the constant variable CHURCH_K1 on the device which can then be used in the agent functions.
 * @param h_CHURCH_K1 value to set the variable
 */
extern void set_CHURCH_K1(unsigned int* h_CHURCH_K1);

extern const unsigned int* get_CHURCH_K1();


extern unsigned int h_env_CHURCH_K1;

/** set_CHURCH_K2
 * Sets the constant variable CHURCH_K2 on the device which can then be used in the agent functions.
 * @param h_CHURCH_K2 value to set the variable
 */
extern void set_CHURCH_K2(unsigned int* h_CHURCH_K2);

extern const unsigned int* get_CHURCH_K2();


extern unsigned int h_env_CHURCH_K2;

/** set_CHURCH_K3
 * Sets the constant variable CHURCH_K3 on the device which can then be used in the agent functions.
 * @param h_CHURCH_K3 value to set the variable
 */
extern void set_CHURCH_K3(unsigned int* h_CHURCH_K3);

extern const unsigned int* get_CHURCH_K3();


extern unsigned int h_env_CHURCH_K3;

/** set_CHURCH_P1
 * Sets the constant variable CHURCH_P1 on the device which can then be used in the agent functions.
 * @param h_CHURCH_P1 value to set the variable
 */
extern void set_CHURCH_P1(float* h_CHURCH_P1);

extern const float* get_CHURCH_P1();


extern float h_env_CHURCH_P1;

/** set_CHURCH_P2
 * Sets the constant variable CHURCH_P2 on the device which can then be used in the agent functions.
 * @param h_CHURCH_P2 value to set the variable
 */
extern void set_CHURCH_P2(float* h_CHURCH_P2);

extern const float* get_CHURCH_P2();


extern float h_env_CHURCH_P2;

/** set_CHURCH_PROB0
 * Sets the constant variable CHURCH_PROB0 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB0 value to set the variable
 */
extern void set_CHURCH_PROB0(float* h_CHURCH_PROB0);

extern const float* get_CHURCH_PROB0();


extern float h_env_CHURCH_PROB0;

/** set_CHURCH_PROB1
 * Sets the constant variable CHURCH_PROB1 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB1 value to set the variable
 */
extern void set_CHURCH_PROB1(float* h_CHURCH_PROB1);

extern const float* get_CHURCH_PROB1();


extern float h_env_CHURCH_PROB1;

/** set_CHURCH_PROB2
 * Sets the constant variable CHURCH_PROB2 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB2 value to set the variable
 */
extern void set_CHURCH_PROB2(float* h_CHURCH_PROB2);

extern const float* get_CHURCH_PROB2();


extern float h_env_CHURCH_PROB2;

/** set_CHURCH_PROB3
 * Sets the constant variable CHURCH_PROB3 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB3 value to set the variable
 */
extern void set_CHURCH_PROB3(float* h_CHURCH_PROB3);

extern const float* get_CHURCH_PROB3();


extern float h_env_CHURCH_PROB3;

/** set_CHURCH_PROB4
 * Sets the constant variable CHURCH_PROB4 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB4 value to set the variable
 */
extern void set_CHURCH_PROB4(float* h_CHURCH_PROB4);

extern const float* get_CHURCH_PROB4();


extern float h_env_CHURCH_PROB4;

/** set_CHURCH_PROB5
 * Sets the constant variable CHURCH_PROB5 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB5 value to set the variable
 */
extern void set_CHURCH_PROB5(float* h_CHURCH_PROB5);

extern const float* get_CHURCH_PROB5();


extern float h_env_CHURCH_PROB5;

/** set_CHURCH_PROB6
 * Sets the constant variable CHURCH_PROB6 on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROB6 value to set the variable
 */
extern void set_CHURCH_PROB6(float* h_CHURCH_PROB6);

extern const float* get_CHURCH_PROB6();


extern float h_env_CHURCH_PROB6;

/** set_CHURCH_DURATION
 * Sets the constant variable CHURCH_DURATION on the device which can then be used in the agent functions.
 * @param h_CHURCH_DURATION value to set the variable
 */
extern void set_CHURCH_DURATION(float* h_CHURCH_DURATION);

extern const float* get_CHURCH_DURATION();


extern float h_env_CHURCH_DURATION;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

