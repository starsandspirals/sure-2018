
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

__constant__ int d_xmachine_memory_HouseholdMembership_count;

__constant__ int d_xmachine_memory_Church_count;

__constant__ int d_xmachine_memory_ChurchMembership_count;

__constant__ int d_xmachine_memory_Transport_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Person_default_count;

__constant__ int d_xmachine_memory_Person_s2_count;

__constant__ int d_xmachine_memory_Household_hhdefault_count;

__constant__ int d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;

__constant__ int d_xmachine_memory_Church_chudefault_count;

__constant__ int d_xmachine_memory_ChurchMembership_chumembershipdefault_count;

__constant__ int d_xmachine_memory_Transport_trdefault_count;


/* Message constants */

/* household_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_household_membership_count;         /**< message list counter*/
__constant__ int d_message_household_membership_output_type;   /**< message output type (single or optional)*/

/* church_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_church_membership_count;         /**< message list counter*/
__constant__ int d_message_church_membership_output_type;   /**< message output type (single or optional)*/

/* location Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_location_count;         /**< message list counter*/
__constant__ int d_message_location_output_type;   /**< message output type (single or optional)*/

	
    
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
		agents_dst->step[output_index] = agents_src->step[index];        
		agents_dst->age[output_index] = agents_src->age[index];        
		agents_dst->gender[output_index] = agents_src->gender[index];        
		agents_dst->householdsize[output_index] = agents_src->householdsize[index];        
		agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];        
		agents_dst->churchdur[output_index] = agents_src->churchdur[index];        
		agents_dst->transportuser[output_index] = agents_src->transportuser[index];        
		agents_dst->transportfreq[output_index] = agents_src->transportfreq[index];        
		agents_dst->transportdur[output_index] = agents_src->transportdur[index];        
		agents_dst->household[output_index] = agents_src->household[index];        
		agents_dst->church[output_index] = agents_src->church[index];        
		agents_dst->busy[output_index] = agents_src->busy[index];        
		agents_dst->startstep[output_index] = agents_src->startstep[index];
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
	    agents_dst->step[output_index] = agents_src->step[index];
	    agents_dst->age[output_index] = agents_src->age[index];
	    agents_dst->gender[output_index] = agents_src->gender[index];
	    agents_dst->householdsize[output_index] = agents_src->householdsize[index];
	    agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];
	    agents_dst->churchdur[output_index] = agents_src->churchdur[index];
	    agents_dst->transportuser[output_index] = agents_src->transportuser[index];
	    agents_dst->transportfreq[output_index] = agents_src->transportfreq[index];
	    agents_dst->transportdur[output_index] = agents_src->transportdur[index];
	    agents_dst->household[output_index] = agents_src->household[index];
	    agents_dst->church[output_index] = agents_src->church[index];
	    agents_dst->busy[output_index] = agents_src->busy[index];
	    agents_dst->startstep[output_index] = agents_src->startstep[index];
    }
}

/** add_Person_agent
 * Continuous Person agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Person_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param step agent variable of type unsigned int
 * @param age agent variable of type unsigned int
 * @param gender agent variable of type unsigned int
 * @param householdsize agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 * @param churchdur agent variable of type unsigned int
 * @param transportuser agent variable of type unsigned int
 * @param transportfreq agent variable of type int
 * @param transportdur agent variable of type int
 * @param household agent variable of type unsigned int
 * @param church agent variable of type unsigned int
 * @param busy agent variable of type unsigned int
 * @param startstep agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int step, unsigned int age, unsigned int gender, unsigned int householdsize, unsigned int churchfreq, unsigned int churchdur, unsigned int transportuser, int transportfreq, int transportdur, unsigned int household, unsigned int church, unsigned int busy, unsigned int startstep){
	
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
	agents->step[index] = step;
	agents->age[index] = age;
	agents->gender[index] = gender;
	agents->householdsize[index] = householdsize;
	agents->churchfreq[index] = churchfreq;
	agents->churchdur[index] = churchdur;
	agents->transportuser[index] = transportuser;
	agents->transportfreq[index] = transportfreq;
	agents->transportdur[index] = transportdur;
	agents->household[index] = household;
	agents->church[index] = church;
	agents->busy[index] = busy;
	agents->startstep[index] = startstep;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int step, unsigned int age, unsigned int gender, unsigned int householdsize, unsigned int churchfreq, unsigned int churchdur, unsigned int transportuser, int transportfreq, int transportdur, unsigned int household, unsigned int church, unsigned int busy, unsigned int startstep){
    add_Person_agent<DISCRETE_2D>(agents, id, step, age, gender, householdsize, churchfreq, churchdur, transportuser, transportfreq, transportdur, household, church, busy, startstep);
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
	ordered_agents->step[index] = unordered_agents->step[old_pos];
	ordered_agents->age[index] = unordered_agents->age[old_pos];
	ordered_agents->gender[index] = unordered_agents->gender[old_pos];
	ordered_agents->householdsize[index] = unordered_agents->householdsize[old_pos];
	ordered_agents->churchfreq[index] = unordered_agents->churchfreq[old_pos];
	ordered_agents->churchdur[index] = unordered_agents->churchdur[old_pos];
	ordered_agents->transportuser[index] = unordered_agents->transportuser[old_pos];
	ordered_agents->transportfreq[index] = unordered_agents->transportfreq[old_pos];
	ordered_agents->transportdur[index] = unordered_agents->transportdur[old_pos];
	ordered_agents->household[index] = unordered_agents->household[old_pos];
	ordered_agents->church[index] = unordered_agents->church[old_pos];
	ordered_agents->busy[index] = unordered_agents->busy[old_pos];
	ordered_agents->startstep[index] = unordered_agents->startstep[old_pos];
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
		agents_dst->step[output_index] = agents_src->step[index];        
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
	    agents_dst->step[output_index] = agents_src->step[index];
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
 * @param step agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param people agent variable of type int
 * @param churchgoing agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 * @param adults agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, unsigned int step, unsigned int size, unsigned int churchgoing, unsigned int churchfreq, unsigned int adults){
	
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
	agents->step[index] = step;
	agents->size[index] = size;
	agents->churchgoing[index] = churchgoing;
	agents->churchfreq[index] = churchfreq;
	agents->adults[index] = adults;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, unsigned int step, unsigned int size, unsigned int churchgoing, unsigned int churchfreq, unsigned int adults){
    add_Household_agent<DISCRETE_2D>(agents, id, step, size, churchgoing, churchfreq, adults);
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
	ordered_agents->step[index] = unordered_agents->step[old_pos];
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
/* Dyanamically created HouseholdMembership agent functions */

/** reset_HouseholdMembership_scan_input
 * HouseholdMembership agent reset scan input function
 * @param agents The xmachine_memory_HouseholdMembership_list agent list
 */
__global__ void reset_HouseholdMembership_scan_input(xmachine_memory_HouseholdMembership_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_HouseholdMembership_Agents
 * HouseholdMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_HouseholdMembership_list agent list destination
 * @param agents_src xmachine_memory_HouseholdMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_HouseholdMembership_Agents(xmachine_memory_HouseholdMembership_list* agents_dst, xmachine_memory_HouseholdMembership_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->household_id[output_index] = agents_src->household_id[index];        
		agents_dst->person_id[output_index] = agents_src->person_id[index];        
		agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];
	}
}

/** append_HouseholdMembership_Agents
 * HouseholdMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_HouseholdMembership_list agent list destination
 * @param agents_src xmachine_memory_HouseholdMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_HouseholdMembership_Agents(xmachine_memory_HouseholdMembership_list* agents_dst, xmachine_memory_HouseholdMembership_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->household_id[output_index] = agents_src->household_id[index];
	    agents_dst->person_id[output_index] = agents_src->person_id[index];
	    agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];
    }
}

/** add_HouseholdMembership_agent
 * Continuous HouseholdMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_HouseholdMembership_list to add agents to 
 * @param household_id agent variable of type unsigned int
 * @param person_id agent variable of type unsigned int
 * @param churchgoing agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_HouseholdMembership_agent(xmachine_memory_HouseholdMembership_list* agents, unsigned int household_id, unsigned int person_id, unsigned int churchgoing){
	
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
	agents->household_id[index] = household_id;
	agents->person_id[index] = person_id;
	agents->churchgoing[index] = churchgoing;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_HouseholdMembership_agent(xmachine_memory_HouseholdMembership_list* agents, unsigned int household_id, unsigned int person_id, unsigned int churchgoing){
    add_HouseholdMembership_agent<DISCRETE_2D>(agents, household_id, person_id, churchgoing);
}

/** reorder_HouseholdMembership_agents
 * Continuous HouseholdMembership agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_HouseholdMembership_agents(unsigned int* values, xmachine_memory_HouseholdMembership_list* unordered_agents, xmachine_memory_HouseholdMembership_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->household_id[index] = unordered_agents->household_id[old_pos];
	ordered_agents->person_id[index] = unordered_agents->person_id[old_pos];
	ordered_agents->churchgoing[index] = unordered_agents->churchgoing[old_pos];
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
		agents_dst->step[output_index] = agents_src->step[index];        
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
	    agents_dst->step[output_index] = agents_src->step[index];
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
 * @param step agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param duration agent variable of type float
 * @param households agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int step, unsigned int size, float duration){
	
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
	agents->step[index] = step;
	agents->size[index] = size;
	agents->duration[index] = duration;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int step, unsigned int size, float duration){
    add_Church_agent<DISCRETE_2D>(agents, id, step, size, duration);
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
	ordered_agents->step[index] = unordered_agents->step[old_pos];
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

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created ChurchMembership agent functions */

/** reset_ChurchMembership_scan_input
 * ChurchMembership agent reset scan input function
 * @param agents The xmachine_memory_ChurchMembership_list agent list
 */
__global__ void reset_ChurchMembership_scan_input(xmachine_memory_ChurchMembership_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_ChurchMembership_Agents
 * ChurchMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_ChurchMembership_list agent list destination
 * @param agents_src xmachine_memory_ChurchMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_ChurchMembership_Agents(xmachine_memory_ChurchMembership_list* agents_dst, xmachine_memory_ChurchMembership_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->church_id[output_index] = agents_src->church_id[index];        
		agents_dst->household_id[output_index] = agents_src->household_id[index];
	}
}

/** append_ChurchMembership_Agents
 * ChurchMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_ChurchMembership_list agent list destination
 * @param agents_src xmachine_memory_ChurchMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_ChurchMembership_Agents(xmachine_memory_ChurchMembership_list* agents_dst, xmachine_memory_ChurchMembership_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->church_id[output_index] = agents_src->church_id[index];
	    agents_dst->household_id[output_index] = agents_src->household_id[index];
    }
}

/** add_ChurchMembership_agent
 * Continuous ChurchMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_ChurchMembership_list to add agents to 
 * @param church_id agent variable of type unsigned int
 * @param household_id agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_ChurchMembership_agent(xmachine_memory_ChurchMembership_list* agents, unsigned int church_id, unsigned int household_id){
	
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
	agents->church_id[index] = church_id;
	agents->household_id[index] = household_id;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_ChurchMembership_agent(xmachine_memory_ChurchMembership_list* agents, unsigned int church_id, unsigned int household_id){
    add_ChurchMembership_agent<DISCRETE_2D>(agents, church_id, household_id);
}

/** reorder_ChurchMembership_agents
 * Continuous ChurchMembership agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_ChurchMembership_agents(unsigned int* values, xmachine_memory_ChurchMembership_list* unordered_agents, xmachine_memory_ChurchMembership_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->church_id[index] = unordered_agents->church_id[old_pos];
	ordered_agents->household_id[index] = unordered_agents->household_id[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Transport agent functions */

/** reset_Transport_scan_input
 * Transport agent reset scan input function
 * @param agents The xmachine_memory_Transport_list agent list
 */
__global__ void reset_Transport_scan_input(xmachine_memory_Transport_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Transport_Agents
 * Transport scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Transport_list agent list destination
 * @param agents_src xmachine_memory_Transport_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Transport_Agents(xmachine_memory_Transport_list* agents_dst, xmachine_memory_Transport_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->step[output_index] = agents_src->step[index];        
		agents_dst->duration[output_index] = agents_src->duration[index];
	}
}

/** append_Transport_Agents
 * Transport scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Transport_list agent list destination
 * @param agents_src xmachine_memory_Transport_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Transport_Agents(xmachine_memory_Transport_list* agents_dst, xmachine_memory_Transport_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->step[output_index] = agents_src->step[index];
	    agents_dst->duration[output_index] = agents_src->duration[index];
    }
}

/** add_Transport_agent
 * Continuous Transport agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Transport_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param step agent variable of type unsigned int
 * @param duration agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Transport_agent(xmachine_memory_Transport_list* agents, unsigned int id, unsigned int step, unsigned int duration){
	
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
	agents->step[index] = step;
	agents->duration[index] = duration;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Transport_agent(xmachine_memory_Transport_list* agents, unsigned int id, unsigned int step, unsigned int duration){
    add_Transport_agent<DISCRETE_2D>(agents, id, step, duration);
}

/** reorder_Transport_agents
 * Continuous Transport agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Transport_agents(unsigned int* values, xmachine_memory_Transport_list* unordered_agents, xmachine_memory_Transport_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->step[index] = unordered_agents->step[old_pos];
	ordered_agents->duration[index] = unordered_agents->duration[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created household_membership message functions */


/** add_household_membership_message
 * Add non partitioned or spatially partitioned household_membership message
 * @param messages xmachine_message_household_membership_list message list to add too
 * @param household_id agent variable of type unsigned int
 * @param person_id agent variable of type unsigned int
 * @param church_id agent variable of type unsigned int
 */
__device__ void add_household_membership_message(xmachine_message_household_membership_list* messages, unsigned int household_id, unsigned int person_id, unsigned int church_id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_household_membership_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_household_membership_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_household_membership_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_household_membership Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->household_id[index] = household_id;
	messages->person_id[index] = person_id;
	messages->church_id[index] = church_id;

}

/**
 * Scatter non partitioned or spatially partitioned household_membership message (for optional messages)
 * @param messages scatter_optional_household_membership_messages Sparse xmachine_message_household_membership_list message list
 * @param message_swap temp xmachine_message_household_membership_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_household_membership_messages(xmachine_message_household_membership_list* messages, xmachine_message_household_membership_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_household_membership_count;

		//AoS - xmachine_message_household_membership Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->household_id[output_index] = messages_swap->household_id[index];
		messages->person_id[output_index] = messages_swap->person_id[index];
		messages->church_id[output_index] = messages_swap->church_id[index];				
	}
}

/** reset_household_membership_swaps
 * Reset non partitioned or spatially partitioned household_membership message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_household_membership_swaps(xmachine_message_household_membership_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_household_membership* get_first_household_membership_message(xmachine_message_household_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_household_membership_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_household_membership Coalesced memory read
	xmachine_message_household_membership temp_message;
	temp_message._position = messages->_position[index];
	temp_message.household_id = messages->household_id[index];
	temp_message.person_id = messages->person_id[index];
	temp_message.church_id = messages->church_id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_household_membership));
	xmachine_message_household_membership* sm_message = ((xmachine_message_household_membership*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_household_membership*)&message_share[d_SM_START]);
}

__device__ xmachine_message_household_membership* get_next_household_membership_message(xmachine_message_household_membership* message, xmachine_message_household_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_household_membership_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_household_membership_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_household_membership Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_household_membership temp_message;
		temp_message._position = messages->_position[index];
		temp_message.household_id = messages->household_id[index];
		temp_message.person_id = messages->person_id[index];
		temp_message.church_id = messages->church_id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_household_membership));
		xmachine_message_household_membership* sm_message = ((xmachine_message_household_membership*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_household_membership));
	return ((xmachine_message_household_membership*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created church_membership message functions */


/** add_church_membership_message
 * Add non partitioned or spatially partitioned church_membership message
 * @param messages xmachine_message_church_membership_list message list to add too
 * @param church_id agent variable of type unsigned int
 * @param household_id agent variable of type unsigned int
 */
__device__ void add_church_membership_message(xmachine_message_church_membership_list* messages, unsigned int church_id, unsigned int household_id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_church_membership_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_church_membership_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_church_membership_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_church_membership Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->church_id[index] = church_id;
	messages->household_id[index] = household_id;

}

/**
 * Scatter non partitioned or spatially partitioned church_membership message (for optional messages)
 * @param messages scatter_optional_church_membership_messages Sparse xmachine_message_church_membership_list message list
 * @param message_swap temp xmachine_message_church_membership_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_church_membership_messages(xmachine_message_church_membership_list* messages, xmachine_message_church_membership_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_church_membership_count;

		//AoS - xmachine_message_church_membership Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->church_id[output_index] = messages_swap->church_id[index];
		messages->household_id[output_index] = messages_swap->household_id[index];				
	}
}

/** reset_church_membership_swaps
 * Reset non partitioned or spatially partitioned church_membership message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_church_membership_swaps(xmachine_message_church_membership_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_church_membership* get_first_church_membership_message(xmachine_message_church_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_church_membership_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_church_membership Coalesced memory read
	xmachine_message_church_membership temp_message;
	temp_message._position = messages->_position[index];
	temp_message.church_id = messages->church_id[index];
	temp_message.household_id = messages->household_id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_church_membership));
	xmachine_message_church_membership* sm_message = ((xmachine_message_church_membership*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_church_membership*)&message_share[d_SM_START]);
}

__device__ xmachine_message_church_membership* get_next_church_membership_message(xmachine_message_church_membership* message, xmachine_message_church_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_church_membership_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_church_membership_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_church_membership Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_church_membership temp_message;
		temp_message._position = messages->_position[index];
		temp_message.church_id = messages->church_id[index];
		temp_message.household_id = messages->household_id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_church_membership));
		xmachine_message_church_membership* sm_message = ((xmachine_message_church_membership*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_church_membership));
	return ((xmachine_message_church_membership*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created location message functions */


/** add_location_message
 * Add non partitioned or spatially partitioned location message
 * @param messages xmachine_message_location_list message list to add too
 * @param person_id agent variable of type unsigned int
 * @param location_type agent variable of type unsigned int
 * @param location_id agent variable of type unsigned int
 * @param day agent variable of type unsigned int
 * @param hour agent variable of type unsigned int
 * @param minute agent variable of type unsigned int
 */
__device__ void add_location_message(xmachine_message_location_list* messages, unsigned int person_id, unsigned int location_type, unsigned int location_id, unsigned int day, unsigned int hour, unsigned int minute){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->person_id[index] = person_id;
	messages->location_type[index] = location_type;
	messages->location_id[index] = location_id;
	messages->day[index] = day;
	messages->hour[index] = hour;
	messages->minute[index] = minute;

}

/**
 * Scatter non partitioned or spatially partitioned location message (for optional messages)
 * @param messages scatter_optional_location_messages Sparse xmachine_message_location_list message list
 * @param message_swap temp xmachine_message_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_location_messages(xmachine_message_location_list* messages, xmachine_message_location_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_location_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->person_id[output_index] = messages_swap->person_id[index];
		messages->location_type[output_index] = messages_swap->location_type[index];
		messages->location_id[output_index] = messages_swap->location_id[index];
		messages->day[output_index] = messages_swap->day[index];
		messages->hour[output_index] = messages_swap->hour[index];
		messages->minute[output_index] = messages_swap->minute[index];				
	}
}

/** reset_location_swaps
 * Reset non partitioned or spatially partitioned location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_location_swaps(xmachine_message_location_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_location* get_first_location_message(xmachine_message_location_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_location_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_location Coalesced memory read
	xmachine_message_location temp_message;
	temp_message._position = messages->_position[index];
	temp_message.person_id = messages->person_id[index];
	temp_message.location_type = messages->location_type[index];
	temp_message.location_id = messages->location_id[index];
	temp_message.day = messages->day[index];
	temp_message.hour = messages->hour[index];
	temp_message.minute = messages->minute[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_location));
	xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_location*)&message_share[d_SM_START]);
}

__device__ xmachine_message_location* get_next_location_message(xmachine_message_location* message, xmachine_message_location_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_location_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_location_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_location Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_location temp_message;
		temp_message._position = messages->_position[index];
		temp_message.person_id = messages->person_id[index];
		temp_message.location_type = messages->location_type[index];
		temp_message.location_id = messages->location_id[index];
		temp_message.day = messages->day[index];
		temp_message.hour = messages->hour[index];
		temp_message.minute = messages->minute[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_location));
		xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_location));
	return ((xmachine_message_location*)&message_share[message_index]);
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
	agent.step = agents->step[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportuser = agents->transportuser[index];
	agent.transportfreq = agents->transportfreq[index];
	agent.transportdur = agents->transportdur[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];

	//FLAME function call
	int dead = !update(&agent, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_update Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportuser[index] = agent.transportuser;
	agents->transportfreq[index] = agent.transportfreq;
	agents->transportdur[index] = agent.transportdur;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
}

/**
 *
 */
__global__ void GPUFLAME_init(xmachine_memory_Person_list* agents, xmachine_message_household_membership_list* household_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_init Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportuser = agents->transportuser[index];
	agent.transportfreq = agents->transportfreq[index];
	agent.transportdur = agents->transportdur[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportuser = 0;
	agent.transportfreq = 0;
	agent.transportdur = 0;
	agent.household = 0;
	agent.church = 0;
	agent.busy = 0;
	agent.startstep = 0;
	}

	//FLAME function call
	int dead = !init(&agent, household_membership_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_init Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportuser[index] = agent.transportuser;
	agents->transportfreq[index] = agent.transportfreq;
	agents->transportdur[index] = agent.transportdur;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	}
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
	agent.step = agents->step[index];
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
	agents->step[index] = agent.step;
	agents->size[index] = agent.size;
	agents->churchgoing[index] = agent.churchgoing;
	agents->churchfreq[index] = agent.churchfreq;
	agents->adults[index] = agent.adults;
}

/**
 *
 */
__global__ void GPUFLAME_hhinit(xmachine_memory_HouseholdMembership_list* agents, xmachine_message_church_membership_list* church_membership_messages, xmachine_message_household_membership_list* household_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_hhinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_HouseholdMembership agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_HouseholdMembership_count){
    
	agent.household_id = agents->household_id[index];
	agent.person_id = agents->person_id[index];
	agent.churchgoing = agents->churchgoing[index];
	} else {
	
	agent.household_id = 0;
	agent.person_id = 0;
	agent.churchgoing = 0;
	}

	//FLAME function call
	int dead = !hhinit(&agent, church_membership_messages, household_membership_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_HouseholdMembership_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_hhinit Coalesced memory write (ignore arrays)
	agents->household_id[index] = agent.household_id;
	agents->person_id[index] = agent.person_id;
	agents->churchgoing[index] = agent.churchgoing;
	}
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
	agent.step = agents->step[index];
	agent.size = agents->size[index];
	agent.duration = agents->duration[index];
    agent.households = &(agents->households[index]);

	//FLAME function call
	int dead = !chuupdate(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_chuupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->size[index] = agent.size;
	agents->duration[index] = agent.duration;
}

/**
 *
 */
__global__ void GPUFLAME_chuinit(xmachine_memory_ChurchMembership_list* agents, xmachine_message_church_membership_list* church_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_ChurchMembership_count)
        return;
    

	//SoA to AoS - xmachine_memory_chuinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_ChurchMembership agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.church_id = agents->church_id[index];
	agent.household_id = agents->household_id[index];

	//FLAME function call
	int dead = !chuinit(&agent, church_membership_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_chuinit Coalesced memory write (ignore arrays)
	agents->church_id[index] = agent.church_id;
	agents->household_id[index] = agent.household_id;
}

/**
 *
 */
__global__ void GPUFLAME_trupdate(xmachine_memory_Transport_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Transport_count)
        return;
    

	//SoA to AoS - xmachine_memory_trupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Transport agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.duration = agents->duration[index];

	//FLAME function call
	int dead = !trupdate(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_trupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
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
