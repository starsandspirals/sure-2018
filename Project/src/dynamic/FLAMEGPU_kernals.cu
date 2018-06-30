
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


#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_Person_count;

__constant__ int d_xmachine_memory_Household_count;

__constant__ int d_xmachine_memory_Church_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Person_default_count;

__constant__ int d_xmachine_memory_Person_s2_count;

__constant__ int d_xmachine_memory_Household_hhdefault_count;

__constant__ int d_xmachine_memory_Church_chudefault_count;


/* Message constants */

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */
    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Person agent functions */

/** reset_Person_scan_input
 * Person agent reset scan input function
 * @param agents The xmachine_memory_Person_list agent list
 */
__global__ void reset_Person_scan_input(xmachine_memory_Person_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Person_Agents
 * Person scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Person_list agent list destination
 * @param agents_src xmachine_memory_Person_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Person_Agents(xmachine_memory_Person_list* agents_dst, xmachine_memory_Person_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->age[output_index] = agents_src->age[index];        
		agents_dst->gender[output_index] = agents_src->gender[index];        
		agents_dst->householdsize[output_index] = agents_src->householdsize[index];
	}
}

/** append_Person_Agents
 * Person scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Person_list agent list destination
 * @param agents_src xmachine_memory_Person_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Person_Agents(xmachine_memory_Person_list* agents_dst, xmachine_memory_Person_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->age[output_index] = agents_src->age[index];
	    agents_dst->gender[output_index] = agents_src->gender[index];
	    agents_dst->householdsize[output_index] = agents_src->householdsize[index];
    }
}

/** add_Person_agent
 * Continuous Person agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Person_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param age agent variable of type unsigned int
 * @param gender agent variable of type unsigned int
 * @param householdsize agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int age, unsigned int gender, unsigned int householdsize){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->age[index] = age;
	agents->gender[index] = gender;
	agents->householdsize[index] = householdsize;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int age, unsigned int gender, unsigned int householdsize){
    add_Person_agent<DISCRETE_2D>(agents, id, age, gender, householdsize);
}

/** reorder_Person_agents
 * Continuous Person agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Person_agents(unsigned int* values, xmachine_memory_Person_list* unordered_agents, xmachine_memory_Person_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->age[index] = unordered_agents->age[old_pos];
	ordered_agents->gender[index] = unordered_agents->gender[old_pos];
	ordered_agents->householdsize[index] = unordered_agents->householdsize[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Household agent functions */

/** reset_Household_scan_input
 * Household agent reset scan input function
 * @param agents The xmachine_memory_Household_list agent list
 */
__global__ void reset_Household_scan_input(xmachine_memory_Household_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Household_Agents
 * Household scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Household_list agent list destination
 * @param agents_src xmachine_memory_Household_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Household_Agents(xmachine_memory_Household_list* agents_dst, xmachine_memory_Household_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->size[output_index] = agents_src->size[index];
	    for (int i=0; i<32; i++){
	      agents_dst->people[(i*xmachine_memory_Household_MAX)+output_index] = agents_src->people[(i*xmachine_memory_Household_MAX)+index];
	    }        
		agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];        
		agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];        
		agents_dst->adults[output_index] = agents_src->adults[index];
	}
}

/** append_Household_Agents
 * Household scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Household_list agent list destination
 * @param agents_src xmachine_memory_Household_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Household_Agents(xmachine_memory_Household_list* agents_dst, xmachine_memory_Household_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->size[output_index] = agents_src->size[index];
	    for (int i=0; i<32; i++){
	      agents_dst->people[(i*xmachine_memory_Household_MAX)+output_index] = agents_src->people[(i*xmachine_memory_Household_MAX)+index];
	    }
	    agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];
	    agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];
	    agents_dst->adults[output_index] = agents_src->adults[index];
    }
}

/** add_Household_agent
 * Continuous Household agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Household_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param people agent variable of type int
 * @param churchgoing agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 * @param adults agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, unsigned int size, unsigned int churchgoing, unsigned int churchfreq, unsigned int adults){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->size[index] = size;
	agents->churchgoing[index] = churchgoing;
	agents->churchfreq[index] = churchfreq;
	agents->adults[index] = adults;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, unsigned int size, unsigned int churchgoing, unsigned int churchfreq, unsigned int adults){
    add_Household_agent<DISCRETE_2D>(agents, id, size, churchgoing, churchfreq, adults);
}

/** reorder_Household_agents
 * Continuous Household agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Household_agents(unsigned int* values, xmachine_memory_Household_list* unordered_agents, xmachine_memory_Household_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->size[index] = unordered_agents->size[old_pos];
	for (int i=0; i<32; i++){
	  ordered_agents->people[(i*xmachine_memory_Household_MAX)+index] = unordered_agents->people[(i*xmachine_memory_Household_MAX)+old_pos];
	}
	ordered_agents->churchgoing[index] = unordered_agents->churchgoing[old_pos];
	ordered_agents->churchfreq[index] = unordered_agents->churchfreq[old_pos];
	ordered_agents->adults[index] = unordered_agents->adults[old_pos];
}

/** get_Household_agent_array_value
 *  Template function for accessing Household agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Household_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_Household_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_Household_agent_array_value
 *  Template function for setting Household agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Household_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_Household_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Church agent functions */

/** reset_Church_scan_input
 * Church agent reset scan input function
 * @param agents The xmachine_memory_Church_list agent list
 */
__global__ void reset_Church_scan_input(xmachine_memory_Church_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Church_Agents
 * Church scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Church_list agent list destination
 * @param agents_src xmachine_memory_Church_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Church_Agents(xmachine_memory_Church_list* agents_dst, xmachine_memory_Church_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->size[output_index] = agents_src->size[index];        
		agents_dst->duration[output_index] = agents_src->duration[index];
	    for (int i=0; i<128; i++){
	      agents_dst->households[(i*xmachine_memory_Church_MAX)+output_index] = agents_src->households[(i*xmachine_memory_Church_MAX)+index];
	    }
	}
}

/** append_Church_Agents
 * Church scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Church_list agent list destination
 * @param agents_src xmachine_memory_Church_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Church_Agents(xmachine_memory_Church_list* agents_dst, xmachine_memory_Church_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->size[output_index] = agents_src->size[index];
	    agents_dst->duration[output_index] = agents_src->duration[index];
	    for (int i=0; i<128; i++){
	      agents_dst->households[(i*xmachine_memory_Church_MAX)+output_index] = agents_src->households[(i*xmachine_memory_Church_MAX)+index];
	    }
    }
}

/** add_Church_agent
 * Continuous Church agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Church_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param duration agent variable of type float
 * @param households agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float duration){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->size[index] = size;
	agents->duration[index] = duration;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float duration){
    add_Church_agent<DISCRETE_2D>(agents, id, size, duration);
}

/** reorder_Church_agents
 * Continuous Church agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Church_agents(unsigned int* values, xmachine_memory_Church_list* unordered_agents, xmachine_memory_Church_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->size[index] = unordered_agents->size[old_pos];
	ordered_agents->duration[index] = unordered_agents->duration[old_pos];
	for (int i=0; i<128; i++){
	  ordered_agents->households[(i*xmachine_memory_Church_MAX)+index] = unordered_agents->households[(i*xmachine_memory_Church_MAX)+old_pos];
	}
}

/** get_Church_agent_array_value
 *  Template function for accessing Church agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Church_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_Church_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_Church_agent_array_value
 *  Template function for setting Church agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Church_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_Church_MAX] = value;
    }
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_update(xmachine_memory_Person_list* agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Person_count)
        return;
    

	//SoA to AoS - xmachine_memory_update Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];

	//FLAME function call
	int dead = !update(&agent, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_update Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
}

/**
 *
 */
__global__ void GPUFLAME_hhupdate(xmachine_memory_Household_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Household_count)
        return;
    

	//SoA to AoS - xmachine_memory_hhupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Household agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.size = agents->size[index];
    agent.people = &(agents->people[index]);
	agent.churchgoing = agents->churchgoing[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.adults = agents->adults[index];

	//FLAME function call
	int dead = !hhupdate(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_hhupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->size[index] = agent.size;
	agents->churchgoing[index] = agent.churchgoing;
	agents->churchfreq[index] = agent.churchfreq;
	agents->adults[index] = agent.adults;
}

/**
 *
 */
__global__ void GPUFLAME_chuupdate(xmachine_memory_Church_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Church_count)
        return;
    

	//SoA to AoS - xmachine_memory_chuupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Church agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.size = agents->size[index];
	agent.duration = agents->duration[index];
    agent.households = &(agents->households[index]);

	//FLAME function call
	int dead = !chuupdate(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_chuupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->size[index] = agent.size;
	agents->duration[index] = agent.duration;
}

	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
