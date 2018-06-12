
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


	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 262144

//Maximum population size of xmachine_memory_prey
#define xmachine_memory_prey_MAX 262144

//Maximum population size of xmachine_memory_predator
#define xmachine_memory_predator_MAX 262144

//Maximum population size of xmachine_memory_grass
#define xmachine_memory_grass_MAX 262144
  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_grass_location
#define xmachine_message_grass_location_MAX 262144

//Maximum population size of xmachine_mmessage_prey_location
#define xmachine_message_prey_location_MAX 262144

//Maximum population size of xmachine_mmessage_pred_location
#define xmachine_message_pred_location_MAX 262144

//Maximum population size of xmachine_mmessage_prey_eaten
#define xmachine_message_prey_eaten_MAX 262144

//Maximum population size of xmachine_mmessage_grass_eaten
#define xmachine_message_grass_eaten_MAX 262144



/* Spatial partitioning grid size definitions */
  
  
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

/** struct xmachine_memory_prey
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_prey
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float type;    /**< X-machine memory variable type of type float.*/
    float fx;    /**< X-machine memory variable fx of type float.*/
    float fy;    /**< X-machine memory variable fy of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    int life;    /**< X-machine memory variable life of type int.*/
};

/** struct xmachine_memory_predator
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_predator
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float type;    /**< X-machine memory variable type of type float.*/
    float fx;    /**< X-machine memory variable fx of type float.*/
    float fy;    /**< X-machine memory variable fy of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    int life;    /**< X-machine memory variable life of type int.*/
};

/** struct xmachine_memory_grass
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_grass
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float type;    /**< X-machine memory variable type of type float.*/
    int dead_cycles;    /**< X-machine memory variable dead_cycles of type int.*/
    int available;    /**< X-machine memory variable available of type int.*/
};



/* Message structures */

/** struct xmachine_message_grass_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_grass_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/
};

/** struct xmachine_message_prey_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_prey_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/
};

/** struct xmachine_message_pred_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pred_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/
};

/** struct xmachine_message_prey_eaten
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_prey_eaten
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int pred_id;        /**< Message variable pred_id of type int.*/
};

/** struct xmachine_message_grass_eaten
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_grass_eaten
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int prey_id;        /**< Message variable prey_id of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_prey_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_prey_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_prey_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_prey_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_prey_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_prey_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_prey_MAX];    /**< X-machine memory variable list y of type float.*/
    float type [xmachine_memory_prey_MAX];    /**< X-machine memory variable list type of type float.*/
    float fx [xmachine_memory_prey_MAX];    /**< X-machine memory variable list fx of type float.*/
    float fy [xmachine_memory_prey_MAX];    /**< X-machine memory variable list fy of type float.*/
    float steer_x [xmachine_memory_prey_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_prey_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    int life [xmachine_memory_prey_MAX];    /**< X-machine memory variable list life of type int.*/
};

/** struct xmachine_memory_predator_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_predator_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_predator_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_predator_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_predator_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_predator_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_predator_MAX];    /**< X-machine memory variable list y of type float.*/
    float type [xmachine_memory_predator_MAX];    /**< X-machine memory variable list type of type float.*/
    float fx [xmachine_memory_predator_MAX];    /**< X-machine memory variable list fx of type float.*/
    float fy [xmachine_memory_predator_MAX];    /**< X-machine memory variable list fy of type float.*/
    float steer_x [xmachine_memory_predator_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_predator_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    int life [xmachine_memory_predator_MAX];    /**< X-machine memory variable list life of type int.*/
};

/** struct xmachine_memory_grass_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_grass_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_grass_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_grass_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_grass_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_grass_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_grass_MAX];    /**< X-machine memory variable list y of type float.*/
    float type [xmachine_memory_grass_MAX];    /**< X-machine memory variable list type of type float.*/
    int dead_cycles [xmachine_memory_grass_MAX];    /**< X-machine memory variable list dead_cycles of type int.*/
    int available [xmachine_memory_grass_MAX];    /**< X-machine memory variable list available of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_grass_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_grass_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_grass_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_grass_location_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_grass_location_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_grass_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_grass_location_MAX];    /**< Message memory variable list y of type float.*/
    
};

/** struct xmachine_message_prey_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_prey_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_prey_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_prey_location_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_prey_location_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_prey_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_prey_location_MAX];    /**< Message memory variable list y of type float.*/
    
};

/** struct xmachine_message_pred_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pred_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pred_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pred_location_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_pred_location_MAX];    /**< Message memory variable list id of type int.*/
    float x [xmachine_message_pred_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pred_location_MAX];    /**< Message memory variable list y of type float.*/
    
};

/** struct xmachine_message_prey_eaten_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_prey_eaten_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_prey_eaten_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_prey_eaten_MAX];  /**< Used during parallel prefix sum */
    
    int pred_id [xmachine_message_prey_eaten_MAX];    /**< Message memory variable list pred_id of type int.*/
    
};

/** struct xmachine_message_grass_eaten_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_grass_eaten_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_grass_eaten_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_grass_eaten_MAX];  /**< Used during parallel prefix sum */
    
    int prey_id [xmachine_message_grass_eaten_MAX];    /**< Message memory variable list prey_id of type int.*/
    
};



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
 * prey_output_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages Pointer to output message list of type xmachine_message_prey_location_list. Must be passed as an argument to the add_prey_location_message function ??.
 */
__FLAME_GPU_FUNC__ int prey_output_location(xmachine_memory_prey* agent, xmachine_message_prey_location_list* prey_location_messages);

/**
 * prey_avoid_pred FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages  pred_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pred_location_message and get_next_pred_location_message functions.
 */
__FLAME_GPU_FUNC__ int prey_avoid_pred(xmachine_memory_prey* agent, xmachine_message_pred_location_list* pred_location_messages);

/**
 * prey_flock FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages  prey_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_location_message and get_next_prey_location_message functions.
 */
__FLAME_GPU_FUNC__ int prey_flock(xmachine_memory_prey* agent, xmachine_message_prey_location_list* prey_location_messages);

/**
 * prey_move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int prey_move(xmachine_memory_prey* agent);

/**
 * prey_eaten FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages  pred_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pred_location_message and get_next_pred_location_message functions.* @param prey_eaten_messages Pointer to output message list of type xmachine_message_prey_eaten_list. Must be passed as an argument to the add_prey_eaten_message function ??.
 */
__FLAME_GPU_FUNC__ int prey_eaten(xmachine_memory_prey* agent, xmachine_message_pred_location_list* pred_location_messages, xmachine_message_prey_eaten_list* prey_eaten_messages);

/**
 * prey_eat_or_starve FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param grass_eaten_messages  grass_eaten_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_grass_eaten_message and get_next_grass_eaten_message functions.
 */
__FLAME_GPU_FUNC__ int prey_eat_or_starve(xmachine_memory_prey* agent, xmachine_message_grass_eaten_list* grass_eaten_messages);

/**
 * prey_reproduction FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_agents Pointer to agent list of type xmachine_memory_prey_list. This must be passed as an argument to the add_prey_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int prey_reproduction(xmachine_memory_prey* agent, xmachine_memory_prey_list* prey_agents, RNG_rand48* rand48);

/**
 * pred_output_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages Pointer to output message list of type xmachine_message_pred_location_list. Must be passed as an argument to the add_pred_location_message function ??.
 */
__FLAME_GPU_FUNC__ int pred_output_location(xmachine_memory_predator* agent, xmachine_message_pred_location_list* pred_location_messages);

/**
 * pred_follow_prey FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages  prey_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_location_message and get_next_prey_location_message functions.
 */
__FLAME_GPU_FUNC__ int pred_follow_prey(xmachine_memory_predator* agent, xmachine_message_prey_location_list* prey_location_messages);

/**
 * pred_avoid FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages  pred_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pred_location_message and get_next_pred_location_message functions.
 */
__FLAME_GPU_FUNC__ int pred_avoid(xmachine_memory_predator* agent, xmachine_message_pred_location_list* pred_location_messages);

/**
 * pred_move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int pred_move(xmachine_memory_predator* agent);

/**
 * pred_eat_or_starve FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param prey_eaten_messages  prey_eaten_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_eaten_message and get_next_prey_eaten_message functions.
 */
__FLAME_GPU_FUNC__ int pred_eat_or_starve(xmachine_memory_predator* agent, xmachine_message_prey_eaten_list* prey_eaten_messages);

/**
 * pred_reproduction FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param predator_agents Pointer to agent list of type xmachine_memory_predator_list. This must be passed as an argument to the add_predator_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int pred_reproduction(xmachine_memory_predator* agent, xmachine_memory_predator_list* predator_agents, RNG_rand48* rand48);

/**
 * grass_output_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param grass_location_messages Pointer to output message list of type xmachine_message_grass_location_list. Must be passed as an argument to the add_grass_location_message function ??.
 */
__FLAME_GPU_FUNC__ int grass_output_location(xmachine_memory_grass* agent, xmachine_message_grass_location_list* grass_location_messages);

/**
 * grass_eaten FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages  prey_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_location_message and get_next_prey_location_message functions.* @param grass_eaten_messages Pointer to output message list of type xmachine_message_grass_eaten_list. Must be passed as an argument to the add_grass_eaten_message function ??.
 */
__FLAME_GPU_FUNC__ int grass_eaten(xmachine_memory_grass* agent, xmachine_message_prey_location_list* prey_location_messages, xmachine_message_grass_eaten_list* grass_eaten_messages);

/**
 * grass_growth FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int grass_growth(xmachine_memory_grass* agent, RNG_rand48* rand48);

  
/* Message Function Prototypes for Brute force (No Partitioning) grass_location message implemented in FLAMEGPU_Kernels */

/** add_grass_location_message
 * Function for all types of message partitioning
 * Adds a new grass_location agent to the xmachine_memory_grass_location_list list using a linear mapping
 * @param agents	xmachine_memory_grass_location_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_grass_location_message(xmachine_message_grass_location_list* grass_location_messages, int id, float x, float y);
 
/** get_first_grass_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param grass_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_grass_location * get_first_grass_location_message(xmachine_message_grass_location_list* grass_location_messages);

/** get_next_grass_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param grass_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_grass_location * get_next_grass_location_message(xmachine_message_grass_location* current, xmachine_message_grass_location_list* grass_location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) prey_location message implemented in FLAMEGPU_Kernels */

/** add_prey_location_message
 * Function for all types of message partitioning
 * Adds a new prey_location agent to the xmachine_memory_prey_location_list list using a linear mapping
 * @param agents	xmachine_memory_prey_location_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_prey_location_message(xmachine_message_prey_location_list* prey_location_messages, int id, float x, float y);
 
/** get_first_prey_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param prey_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_prey_location * get_first_prey_location_message(xmachine_message_prey_location_list* prey_location_messages);

/** get_next_prey_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param prey_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_prey_location * get_next_prey_location_message(xmachine_message_prey_location* current, xmachine_message_prey_location_list* prey_location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) pred_location message implemented in FLAMEGPU_Kernels */

/** add_pred_location_message
 * Function for all types of message partitioning
 * Adds a new pred_location agent to the xmachine_memory_pred_location_list list using a linear mapping
 * @param agents	xmachine_memory_pred_location_list agent list
 * @param id	message variable of type int
 * @param x	message variable of type float
 * @param y	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_pred_location_message(xmachine_message_pred_location_list* pred_location_messages, int id, float x, float y);
 
/** get_first_pred_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param pred_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pred_location * get_first_pred_location_message(xmachine_message_pred_location_list* pred_location_messages);

/** get_next_pred_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param pred_location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pred_location * get_next_pred_location_message(xmachine_message_pred_location* current, xmachine_message_pred_location_list* pred_location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) prey_eaten message implemented in FLAMEGPU_Kernels */

/** add_prey_eaten_message
 * Function for all types of message partitioning
 * Adds a new prey_eaten agent to the xmachine_memory_prey_eaten_list list using a linear mapping
 * @param agents	xmachine_memory_prey_eaten_list agent list
 * @param pred_id	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_prey_eaten_message(xmachine_message_prey_eaten_list* prey_eaten_messages, int pred_id);
 
/** get_first_prey_eaten_message
 * Get first message function for non partitioned (brute force) messages
 * @param prey_eaten_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_prey_eaten * get_first_prey_eaten_message(xmachine_message_prey_eaten_list* prey_eaten_messages);

/** get_next_prey_eaten_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param prey_eaten_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_prey_eaten * get_next_prey_eaten_message(xmachine_message_prey_eaten* current, xmachine_message_prey_eaten_list* prey_eaten_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) grass_eaten message implemented in FLAMEGPU_Kernels */

/** add_grass_eaten_message
 * Function for all types of message partitioning
 * Adds a new grass_eaten agent to the xmachine_memory_grass_eaten_list list using a linear mapping
 * @param agents	xmachine_memory_grass_eaten_list agent list
 * @param prey_id	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_grass_eaten_message(xmachine_message_grass_eaten_list* grass_eaten_messages, int prey_id);
 
/** get_first_grass_eaten_message
 * Get first message function for non partitioned (brute force) messages
 * @param grass_eaten_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_grass_eaten * get_first_grass_eaten_message(xmachine_message_grass_eaten_list* grass_eaten_messages);

/** get_next_grass_eaten_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param grass_eaten_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_grass_eaten * get_next_grass_eaten_message(xmachine_message_grass_eaten* current, xmachine_message_grass_eaten_list* grass_eaten_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_prey_agent
 * Adds a new continuous valued prey agent to the xmachine_memory_prey_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_prey_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param type	agent agent variable of type float
 * @param fx	agent agent variable of type float
 * @param fy	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param life	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_prey_agent(xmachine_memory_prey_list* agents, int id, float x, float y, float type, float fx, float fy, float steer_x, float steer_y, int life);

/** add_predator_agent
 * Adds a new continuous valued predator agent to the xmachine_memory_predator_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_predator_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param type	agent agent variable of type float
 * @param fx	agent agent variable of type float
 * @param fy	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param life	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_predator_agent(xmachine_memory_predator_list* agents, int id, float x, float y, float type, float fx, float fy, float steer_x, float steer_y, int life);

/** add_grass_agent
 * Adds a new continuous valued grass agent to the xmachine_memory_grass_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_grass_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param type	agent agent variable of type float
 * @param dead_cycles	agent agent variable of type int
 * @param available	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_grass_agent(xmachine_memory_grass_list* agents, int id, float x, float y, float type, int dead_cycles, int available);


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
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
 * @param h_preys Pointer to agent list on the host
 * @param d_preys Pointer to agent list on the GPU device
 * @param h_xmachine_memory_prey_count Pointer to agent counter
 * @param h_predators Pointer to agent list on the host
 * @param d_predators Pointer to agent list on the GPU device
 * @param h_xmachine_memory_predator_count Pointer to agent counter
 * @param h_grasss Pointer to agent list on the host
 * @param d_grasss Pointer to agent list on the GPU device
 * @param h_xmachine_memory_grass_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_prey_list* h_preys_default1, xmachine_memory_prey_list* d_preys_default1, int h_xmachine_memory_prey_default1_count,xmachine_memory_predator_list* h_predators_default2, xmachine_memory_predator_list* d_predators_default2, int h_xmachine_memory_predator_default2_count,xmachine_memory_grass_list* h_grasss_default3, xmachine_memory_grass_list* d_grasss_default3, int h_xmachine_memory_grass_default3_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_preys Pointer to agent list on the host
 * @param h_xmachine_memory_prey_count Pointer to agent counter
 * @param h_predators Pointer to agent list on the host
 * @param h_xmachine_memory_predator_count Pointer to agent counter
 * @param h_grasss Pointer to agent list on the host
 * @param h_xmachine_memory_grass_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_prey_list* h_preys, int* h_xmachine_memory_prey_count,xmachine_memory_predator_list* h_predators, int* h_xmachine_memory_predator_count,xmachine_memory_grass_list* h_grasss, int* h_xmachine_memory_grass_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_prey_MAX_count
 * Gets the max agent count for the prey agent type 
 * @return		the maximum prey agent count
 */
extern int get_agent_prey_MAX_count();



/** get_agent_prey_default1_count
 * Gets the agent count for the prey agent type in state default1
 * @return		the current prey agent count in state default1
 */
extern int get_agent_prey_default1_count();

/** reset_default1_count
 * Resets the agent count of the prey in state default1 to 0. This is useful for interacting with some visualisations.
 */
extern void reset_prey_default1_count();

/** get_device_prey_default1_agents
 * Gets a pointer to xmachine_memory_prey_list on the GPU device
 * @return		a xmachine_memory_prey_list on the GPU device
 */
extern xmachine_memory_prey_list* get_device_prey_default1_agents();

/** get_host_prey_default1_agents
 * Gets a pointer to xmachine_memory_prey_list on the CPU host
 * @return		a xmachine_memory_prey_list on the CPU host
 */
extern xmachine_memory_prey_list* get_host_prey_default1_agents();


/** sort_preys_default1
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_preys_default1(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_prey_list* agents));


    
/** get_agent_predator_MAX_count
 * Gets the max agent count for the predator agent type 
 * @return		the maximum predator agent count
 */
extern int get_agent_predator_MAX_count();



/** get_agent_predator_default2_count
 * Gets the agent count for the predator agent type in state default2
 * @return		the current predator agent count in state default2
 */
extern int get_agent_predator_default2_count();

/** reset_default2_count
 * Resets the agent count of the predator in state default2 to 0. This is useful for interacting with some visualisations.
 */
extern void reset_predator_default2_count();

/** get_device_predator_default2_agents
 * Gets a pointer to xmachine_memory_predator_list on the GPU device
 * @return		a xmachine_memory_predator_list on the GPU device
 */
extern xmachine_memory_predator_list* get_device_predator_default2_agents();

/** get_host_predator_default2_agents
 * Gets a pointer to xmachine_memory_predator_list on the CPU host
 * @return		a xmachine_memory_predator_list on the CPU host
 */
extern xmachine_memory_predator_list* get_host_predator_default2_agents();


/** sort_predators_default2
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_predators_default2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_predator_list* agents));


    
/** get_agent_grass_MAX_count
 * Gets the max agent count for the grass agent type 
 * @return		the maximum grass agent count
 */
extern int get_agent_grass_MAX_count();



/** get_agent_grass_default3_count
 * Gets the agent count for the grass agent type in state default3
 * @return		the current grass agent count in state default3
 */
extern int get_agent_grass_default3_count();

/** reset_default3_count
 * Resets the agent count of the grass in state default3 to 0. This is useful for interacting with some visualisations.
 */
extern void reset_grass_default3_count();

/** get_device_grass_default3_agents
 * Gets a pointer to xmachine_memory_grass_list on the GPU device
 * @return		a xmachine_memory_grass_list on the GPU device
 */
extern xmachine_memory_grass_list* get_device_grass_default3_agents();

/** get_host_grass_default3_agents
 * Gets a pointer to xmachine_memory_grass_list on the CPU host
 * @return		a xmachine_memory_grass_list on the CPU host
 */
extern xmachine_memory_grass_list* get_host_grass_default3_agents();


/** sort_grasss_default3
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_grasss_default3(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_grass_list* agents));


  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_prey_default1_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_prey_default1_id_variable();

/** int count_prey_default1_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_prey_default1_id_variable(int count_value);

/** float reduce_prey_default1_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_x_variable();

/** float reduce_prey_default1_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_y_variable();

/** float reduce_prey_default1_type_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_type_variable();

/** float reduce_prey_default1_fx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_fx_variable();

/** float reduce_prey_default1_fy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_fy_variable();

/** float reduce_prey_default1_steer_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_steer_x_variable();

/** float reduce_prey_default1_steer_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_prey_default1_steer_y_variable();

/** int reduce_prey_default1_life_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_prey_default1_life_variable();

/** int count_prey_default1_life_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_prey_default1_life_variable(int count_value);

/** int reduce_predator_default2_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_predator_default2_id_variable();

/** int count_predator_default2_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_predator_default2_id_variable(int count_value);

/** float reduce_predator_default2_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_x_variable();

/** float reduce_predator_default2_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_y_variable();

/** float reduce_predator_default2_type_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_type_variable();

/** float reduce_predator_default2_fx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_fx_variable();

/** float reduce_predator_default2_fy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_fy_variable();

/** float reduce_predator_default2_steer_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_steer_x_variable();

/** float reduce_predator_default2_steer_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_predator_default2_steer_y_variable();

/** int reduce_predator_default2_life_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_predator_default2_life_variable();

/** int count_predator_default2_life_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_predator_default2_life_variable(int count_value);

/** int reduce_grass_default3_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_grass_default3_id_variable();

/** int count_grass_default3_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_grass_default3_id_variable(int count_value);

/** float reduce_grass_default3_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_grass_default3_x_variable();

/** float reduce_grass_default3_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_grass_default3_y_variable();

/** float reduce_grass_default3_type_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_grass_default3_type_variable();

/** int reduce_grass_default3_dead_cycles_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_grass_default3_dead_cycles_variable();

/** int count_grass_default3_dead_cycles_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_grass_default3_dead_cycles_variable(int count_value);

/** int reduce_grass_default3_available_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_grass_default3_available_variable();

/** int count_grass_default3_available_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_grass_default3_available_variable(int count_value);


  
/* global constant variables */

__constant__ float REPRODUCE_PREY_PROB;

__constant__ float REPRODUCE_PREDATOR_PROB;

__constant__ int GAIN_FROM_FOOD_PREDATOR;

__constant__ int GAIN_FROM_FOOD_PREY;

__constant__ int GRASS_REGROW_CYCLES;

/** set_REPRODUCE_PREY_PROB
 * Sets the constant variable REPRODUCE_PREY_PROB on the device which can then be used in the agent functions.
 * @param h_REPRODUCE_PREY_PROB value to set the variable
 */
extern void set_REPRODUCE_PREY_PROB(float* h_REPRODUCE_PREY_PROB);


extern const float* get_REPRODUCE_PREY_PROB();


extern float h_env_REPRODUCE_PREY_PROB;

/** set_REPRODUCE_PREDATOR_PROB
 * Sets the constant variable REPRODUCE_PREDATOR_PROB on the device which can then be used in the agent functions.
 * @param h_REPRODUCE_PREDATOR_PROB value to set the variable
 */
extern void set_REPRODUCE_PREDATOR_PROB(float* h_REPRODUCE_PREDATOR_PROB);


extern const float* get_REPRODUCE_PREDATOR_PROB();


extern float h_env_REPRODUCE_PREDATOR_PROB;

/** set_GAIN_FROM_FOOD_PREDATOR
 * Sets the constant variable GAIN_FROM_FOOD_PREDATOR on the device which can then be used in the agent functions.
 * @param h_GAIN_FROM_FOOD_PREDATOR value to set the variable
 */
extern void set_GAIN_FROM_FOOD_PREDATOR(int* h_GAIN_FROM_FOOD_PREDATOR);


extern const int* get_GAIN_FROM_FOOD_PREDATOR();


extern int h_env_GAIN_FROM_FOOD_PREDATOR;

/** set_GAIN_FROM_FOOD_PREY
 * Sets the constant variable GAIN_FROM_FOOD_PREY on the device which can then be used in the agent functions.
 * @param h_GAIN_FROM_FOOD_PREY value to set the variable
 */
extern void set_GAIN_FROM_FOOD_PREY(int* h_GAIN_FROM_FOOD_PREY);


extern const int* get_GAIN_FROM_FOOD_PREY();


extern int h_env_GAIN_FROM_FOOD_PREY;

/** set_GRASS_REGROW_CYCLES
 * Sets the constant variable GRASS_REGROW_CYCLES on the device which can then be used in the agent functions.
 * @param h_GRASS_REGROW_CYCLES value to set the variable
 */
extern void set_GRASS_REGROW_CYCLES(int* h_GRASS_REGROW_CYCLES);


extern const int* get_GRASS_REGROW_CYCLES();


extern int h_env_GRASS_REGROW_CYCLES;


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

#endif //__HEADER

