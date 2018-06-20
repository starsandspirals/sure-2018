
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
#define buffer_size_MAX 256

//Maximum population size of xmachine_memory_Agent
#define xmachine_memory_Agent_MAX 256 
//Agent variable array length for xmachine_memory_Agent->example_array
#define xmachine_memory_Agent_example_array_LENGTH 4


  
  
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

/** struct xmachine_memory_Agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Agent
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int age;    /**< X-machine memory variable age of type unsigned int.*/
    float *example_array;    /**< X-machine memory variable example_array of type float.*/
    ivec4 example_vector;    /**< X-machine memory variable example_vector of type ivec4.*/
    unsigned int dead;    /**< X-machine memory variable dead of type unsigned int.*/
};



/* Message structures */



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Agent_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int age [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list age of type unsigned int.*/
    float example_array [xmachine_memory_Agent_MAX*4];    /**< X-machine memory variable list example_array of type float.*/
    ivec4 example_vector [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list example_vector of type ivec4.*/
    unsigned int dead [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list dead of type unsigned int.*/
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
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int update(xmachine_memory_Agent* agent, RNG_rand48* rand48);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Agent_agent
 * Adds a new continuous valued Agent agent to the xmachine_memory_Agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Agent_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param age	agent agent variable of type unsigned int
 * @param example_vector	agent agent variable of type ivec4
 * @param dead	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int age, ivec4 example_vector, unsigned int dead);

/** get_Agent_agent_array_value
 *  Template function for accessing Agent agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Agent_agent_array_value(T *array, unsigned int index);

/** set_Agent_agent_array_value
 *  Template function for setting Agent agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Agent_agent_array_value(T *array, unsigned int index, T value);


  


  
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
 * @param h_Agents Pointer to agent list on the host
 * @param d_Agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Agent_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Agent_list* h_Agents_default, xmachine_memory_Agent_list* d_Agents_default, int h_xmachine_memory_Agent_default_count,xmachine_memory_Agent_list* h_Agents_s2, xmachine_memory_Agent_list* d_Agents_s2, int h_xmachine_memory_Agent_s2_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Agents Pointer to agent list on the host
 * @param h_xmachine_memory_Agent_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_Agent_list* h_Agents, int* h_xmachine_memory_Agent_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Agent_MAX_count
 * Gets the max agent count for the Agent agent type 
 * @return		the maximum Agent agent count
 */
extern int get_agent_Agent_MAX_count();



/** get_agent_Agent_default_count
 * Gets the agent count for the Agent agent type in state default
 * @return		the current Agent agent count in state default
 */
extern int get_agent_Agent_default_count();

/** reset_default_count
 * Resets the agent count of the Agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Agent_default_count();

/** get_device_Agent_default_agents
 * Gets a pointer to xmachine_memory_Agent_list on the GPU device
 * @return		a xmachine_memory_Agent_list on the GPU device
 */
extern xmachine_memory_Agent_list* get_device_Agent_default_agents();

/** get_host_Agent_default_agents
 * Gets a pointer to xmachine_memory_Agent_list on the CPU host
 * @return		a xmachine_memory_Agent_list on the CPU host
 */
extern xmachine_memory_Agent_list* get_host_Agent_default_agents();


/** sort_Agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Agent_list* agents));


/** get_agent_Agent_s2_count
 * Gets the agent count for the Agent agent type in state s2
 * @return		the current Agent agent count in state s2
 */
extern int get_agent_Agent_s2_count();

/** reset_s2_count
 * Resets the agent count of the Agent in state s2 to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Agent_s2_count();

/** get_device_Agent_s2_agents
 * Gets a pointer to xmachine_memory_Agent_list on the GPU device
 * @return		a xmachine_memory_Agent_list on the GPU device
 */
extern xmachine_memory_Agent_list* get_device_Agent_s2_agents();

/** get_host_Agent_s2_agents
 * Gets a pointer to xmachine_memory_Agent_list on the CPU host
 * @return		a xmachine_memory_Agent_list on the CPU host
 */
extern xmachine_memory_Agent_list* get_host_Agent_s2_agents();


/** sort_Agents_s2
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Agents_s2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Agent_list* agents));



/* Host based access of agent variables*/

/** unsigned int get_Agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Agent_default_variable_id(unsigned int index);

/** unsigned int get_Agent_default_variable_age(unsigned int index)
 * Gets the value of the age variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Agent_default_variable_age(unsigned int index);

/** float get_Agent_default_variable_example_array(unsigned int index, unsigned int element)
 * Gets the element-th value of the example_array variable array of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable example_array
 */
__host__ float get_Agent_default_variable_example_array(unsigned int index, unsigned int element);

/** ivec4 get_Agent_default_variable_example_vector(unsigned int index)
 * Gets the value of the example_vector variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable example_vector
 */
__host__ ivec4 get_Agent_default_variable_example_vector(unsigned int index);

/** unsigned int get_Agent_default_variable_dead(unsigned int index)
 * Gets the value of the dead variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dead
 */
__host__ unsigned int get_Agent_default_variable_dead(unsigned int index);

/** unsigned int get_Agent_s2_variable_id(unsigned int index)
 * Gets the value of the id variable of an Agent agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Agent_s2_variable_id(unsigned int index);

/** unsigned int get_Agent_s2_variable_age(unsigned int index)
 * Gets the value of the age variable of an Agent agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable age
 */
__host__ unsigned int get_Agent_s2_variable_age(unsigned int index);

/** float get_Agent_s2_variable_example_array(unsigned int index, unsigned int element)
 * Gets the element-th value of the example_array variable array of an Agent agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable example_array
 */
__host__ float get_Agent_s2_variable_example_array(unsigned int index, unsigned int element);

/** ivec4 get_Agent_s2_variable_example_vector(unsigned int index)
 * Gets the value of the example_vector variable of an Agent agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable example_vector
 */
__host__ ivec4 get_Agent_s2_variable_example_vector(unsigned int index);

/** unsigned int get_Agent_s2_variable_dead(unsigned int index)
 * Gets the value of the dead variable of an Agent agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dead
 */
__host__ unsigned int get_Agent_s2_variable_dead(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_Agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Agent struct.
 */
xmachine_memory_Agent* h_allocate_agent_Agent();
/** h_free_agent_Agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Agent(xmachine_memory_Agent** agent);
/** h_allocate_agent_Agent_array
 * Utility function to allocate an array of structs for  Agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Agent** h_allocate_agent_Agent_array(unsigned int count);
/** h_free_agent_Agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Agent_array(xmachine_memory_Agent*** agents, unsigned int count);


/** h_add_agent_Agent_default
 * Host function to add a single agent of type Agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Agent_default(xmachine_memory_Agent* agent);

/** h_add_agents_Agent_default(
 * Host function to add multiple agents of type Agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Agent_default(xmachine_memory_Agent** agents, unsigned int count);


/** h_add_agent_Agent_s2
 * Host function to add a single agent of type Agent to the s2 state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Agent_s2 instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Agent_s2(xmachine_memory_Agent* agent);

/** h_add_agents_Agent_s2(
 * Host function to add multiple agents of type Agent to the s2 state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Agent_s2(xmachine_memory_Agent** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** unsigned int reduce_Agent_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_id_variable();



/** unsigned int count_Agent_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_id_variable(int count_value);

/** unsigned int min_Agent_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_id_variable();
/** unsigned int max_Agent_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_id_variable();

/** unsigned int reduce_Agent_default_age_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_age_variable();



/** unsigned int count_Agent_default_age_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_age_variable(int count_value);

/** unsigned int min_Agent_default_age_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_age_variable();
/** unsigned int max_Agent_default_age_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_age_variable();

/** ivec4 reduce_Agent_default_example_vector_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
ivec4 reduce_Agent_default_example_vector_variable();



/** unsigned int reduce_Agent_default_dead_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_dead_variable();



/** unsigned int count_Agent_default_dead_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_dead_variable(int count_value);

/** unsigned int min_Agent_default_dead_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_dead_variable();
/** unsigned int max_Agent_default_dead_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_dead_variable();

/** unsigned int reduce_Agent_s2_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_s2_id_variable();



/** unsigned int count_Agent_s2_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_s2_id_variable(int count_value);

/** unsigned int min_Agent_s2_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_s2_id_variable();
/** unsigned int max_Agent_s2_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_s2_id_variable();

/** unsigned int reduce_Agent_s2_age_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_s2_age_variable();



/** unsigned int count_Agent_s2_age_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_s2_age_variable(int count_value);

/** unsigned int min_Agent_s2_age_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_s2_age_variable();
/** unsigned int max_Agent_s2_age_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_s2_age_variable();

/** ivec4 reduce_Agent_s2_example_vector_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
ivec4 reduce_Agent_s2_example_vector_variable();



/** unsigned int reduce_Agent_s2_dead_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_s2_dead_variable();



/** unsigned int count_Agent_s2_dead_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_s2_dead_variable(int count_value);

/** unsigned int min_Agent_s2_dead_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_s2_dead_variable();
/** unsigned int max_Agent_s2_dead_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_s2_dead_variable();


  
/* global constant variables */

__constant__ float PROB_DEATH;

__constant__ unsigned int SCALE_FACTOR;

__constant__ unsigned int MAX_AGE;

/** set_PROB_DEATH
 * Sets the constant variable PROB_DEATH on the device which can then be used in the agent functions.
 * @param h_PROB_DEATH value to set the variable
 */
extern void set_PROB_DEATH(float* h_PROB_DEATH);

extern const float* get_PROB_DEATH();


extern float h_env_PROB_DEATH;

/** set_SCALE_FACTOR
 * Sets the constant variable SCALE_FACTOR on the device which can then be used in the agent functions.
 * @param h_SCALE_FACTOR value to set the variable
 */
extern void set_SCALE_FACTOR(unsigned int* h_SCALE_FACTOR);

extern const unsigned int* get_SCALE_FACTOR();


extern unsigned int h_env_SCALE_FACTOR;

/** set_MAX_AGE
 * Sets the constant variable MAX_AGE on the device which can then be used in the agent functions.
 * @param h_MAX_AGE value to set the variable
 */
extern void set_MAX_AGE(unsigned int* h_MAX_AGE);

extern const unsigned int* get_MAX_AGE();


extern unsigned int h_env_MAX_AGE;


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

