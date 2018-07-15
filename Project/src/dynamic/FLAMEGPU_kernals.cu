
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

__constant__ int d_xmachine_memory_TBAssignment_count;

__constant__ int d_xmachine_memory_Household_count;

__constant__ int d_xmachine_memory_HouseholdMembership_count;

__constant__ int d_xmachine_memory_Church_count;

__constant__ int d_xmachine_memory_ChurchMembership_count;

__constant__ int d_xmachine_memory_Transport_count;

__constant__ int d_xmachine_memory_TransportMembership_count;

__constant__ int d_xmachine_memory_Clinic_count;

__constant__ int d_xmachine_memory_Workplace_count;

__constant__ int d_xmachine_memory_WorkplaceMembership_count;

__constant__ int d_xmachine_memory_Bar_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Person_default_count;

__constant__ int d_xmachine_memory_Person_s2_count;

__constant__ int d_xmachine_memory_TBAssignment_tbdefault_count;

__constant__ int d_xmachine_memory_Household_hhdefault_count;

__constant__ int d_xmachine_memory_HouseholdMembership_hhmembershipdefault_count;

__constant__ int d_xmachine_memory_Church_chudefault_count;

__constant__ int d_xmachine_memory_ChurchMembership_chumembershipdefault_count;

__constant__ int d_xmachine_memory_Transport_trdefault_count;

__constant__ int d_xmachine_memory_TransportMembership_trmembershipdefault_count;

__constant__ int d_xmachine_memory_Clinic_cldefault_count;

__constant__ int d_xmachine_memory_Workplace_wpdefault_count;

__constant__ int d_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count;

__constant__ int d_xmachine_memory_Bar_bdefault_count;


/* Message constants */

/* tb_assignment Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_tb_assignment_count;         /**< message list counter*/
__constant__ int d_message_tb_assignment_output_type;   /**< message output type (single or optional)*/

/* household_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_household_membership_count;         /**< message list counter*/
__constant__ int d_message_household_membership_output_type;   /**< message output type (single or optional)*/

/* church_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_church_membership_count;         /**< message list counter*/
__constant__ int d_message_church_membership_output_type;   /**< message output type (single or optional)*/

/* transport_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_transport_membership_count;         /**< message list counter*/
__constant__ int d_message_transport_membership_output_type;   /**< message output type (single or optional)*/

/* workplace_membership Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_workplace_membership_count;         /**< message list counter*/
__constant__ int d_message_workplace_membership_output_type;   /**< message output type (single or optional)*/

/* location Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_location_count;         /**< message list counter*/
__constant__ int d_message_location_output_type;   /**< message output type (single or optional)*/

/* infection Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_infection_count;         /**< message list counter*/
__constant__ int d_message_infection_output_type;   /**< message output type (single or optional)*/

	
    
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


/** infect_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Person_list representing agent i the current state
 * @param nextState xmachine_memory_Person_list representing agent i the next state
 */
 __global__ void infect_function_filter(xmachine_memory_Person_list* currentState, xmachine_memory_Person_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Person_count){
	
		//apply the filter
		if ((((currentState->lambda[index]==0.0)||(currentState->location[index]==7))||(currentState->activetb[index]==1))==0)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->step[index] = currentState->step[index];
			nextState->householdtime[index] = currentState->householdtime[index];
			nextState->churchtime[index] = currentState->churchtime[index];
			nextState->transporttime[index] = currentState->transporttime[index];
			nextState->clinictime[index] = currentState->clinictime[index];
			nextState->workplacetime[index] = currentState->workplacetime[index];
			nextState->outsidetime[index] = currentState->outsidetime[index];
			nextState->age[index] = currentState->age[index];
			nextState->gender[index] = currentState->gender[index];
			nextState->householdsize[index] = currentState->householdsize[index];
			nextState->churchfreq[index] = currentState->churchfreq[index];
			nextState->churchdur[index] = currentState->churchdur[index];
			nextState->transportdur[index] = currentState->transportdur[index];
			nextState->transportday1[index] = currentState->transportday1[index];
			nextState->transportday2[index] = currentState->transportday2[index];
			nextState->household[index] = currentState->household[index];
			nextState->church[index] = currentState->church[index];
			nextState->transport[index] = currentState->transport[index];
			nextState->workplace[index] = currentState->workplace[index];
			nextState->busy[index] = currentState->busy[index];
			nextState->startstep[index] = currentState->startstep[index];
			nextState->location[index] = currentState->location[index];
			nextState->locationid[index] = currentState->locationid[index];
			nextState->hiv[index] = currentState->hiv[index];
			nextState->art[index] = currentState->art[index];
			nextState->activetb[index] = currentState->activetb[index];
			nextState->artday[index] = currentState->artday[index];
			nextState->p[index] = currentState->p[index];
			nextState->q[index] = currentState->q[index];
			nextState->infections[index] = currentState->infections[index];
			nextState->lastinfected[index] = currentState->lastinfected[index];
			nextState->lastinfectedid[index] = currentState->lastinfectedid[index];
			nextState->time_step[index] = currentState->time_step[index];
			nextState->lambda[index] = currentState->lambda[index];
			nextState->timevisiting[index] = currentState->timevisiting[index];
			nextState->bargoing[index] = currentState->bargoing[index];
			nextState->barday[index] = currentState->barday[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** hhupdate_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Household_list representing agent i the current state
 * @param nextState xmachine_memory_Household_list representing agent i the next state
 */
 __global__ void hhupdate_function_filter(xmachine_memory_Household_list* currentState, xmachine_memory_Household_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Household_count){
	
		//apply the filter
		if (currentState->active[index]==1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->lambda[index] = currentState->lambda[index];
			nextState->active[index] = currentState->active[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** chuupdate_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Church_list representing agent i the current state
 * @param nextState xmachine_memory_Church_list representing agent i the next state
 */
 __global__ void chuupdate_function_filter(xmachine_memory_Church_list* currentState, xmachine_memory_Church_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Church_count){
	
		//apply the filter
		if (currentState->active[index]==1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->size[index] = currentState->size[index];
			nextState->lambda[index] = currentState->lambda[index];
			nextState->active[index] = currentState->active[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
 }

/** trupdate_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Transport_list representing agent i the current state
 * @param nextState xmachine_memory_Transport_list representing agent i the next state
 */
 __global__ void trupdate_function_filter(xmachine_memory_Transport_list* currentState, xmachine_memory_Transport_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Transport_count){
	
		//apply the filter
		if (currentState->active[index]==1)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->lambda[index] = currentState->lambda[index];
			nextState->active[index] = currentState->active[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
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
		agents_dst->householdtime[output_index] = agents_src->householdtime[index];        
		agents_dst->churchtime[output_index] = agents_src->churchtime[index];        
		agents_dst->transporttime[output_index] = agents_src->transporttime[index];        
		agents_dst->clinictime[output_index] = agents_src->clinictime[index];        
		agents_dst->workplacetime[output_index] = agents_src->workplacetime[index];        
		agents_dst->outsidetime[output_index] = agents_src->outsidetime[index];        
		agents_dst->age[output_index] = agents_src->age[index];        
		agents_dst->gender[output_index] = agents_src->gender[index];        
		agents_dst->householdsize[output_index] = agents_src->householdsize[index];        
		agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];        
		agents_dst->churchdur[output_index] = agents_src->churchdur[index];        
		agents_dst->transportdur[output_index] = agents_src->transportdur[index];        
		agents_dst->transportday1[output_index] = agents_src->transportday1[index];        
		agents_dst->transportday2[output_index] = agents_src->transportday2[index];        
		agents_dst->household[output_index] = agents_src->household[index];        
		agents_dst->church[output_index] = agents_src->church[index];        
		agents_dst->transport[output_index] = agents_src->transport[index];        
		agents_dst->workplace[output_index] = agents_src->workplace[index];        
		agents_dst->busy[output_index] = agents_src->busy[index];        
		agents_dst->startstep[output_index] = agents_src->startstep[index];        
		agents_dst->location[output_index] = agents_src->location[index];        
		agents_dst->locationid[output_index] = agents_src->locationid[index];        
		agents_dst->hiv[output_index] = agents_src->hiv[index];        
		agents_dst->art[output_index] = agents_src->art[index];        
		agents_dst->activetb[output_index] = agents_src->activetb[index];        
		agents_dst->artday[output_index] = agents_src->artday[index];        
		agents_dst->p[output_index] = agents_src->p[index];        
		agents_dst->q[output_index] = agents_src->q[index];        
		agents_dst->infections[output_index] = agents_src->infections[index];        
		agents_dst->lastinfected[output_index] = agents_src->lastinfected[index];        
		agents_dst->lastinfectedid[output_index] = agents_src->lastinfectedid[index];        
		agents_dst->time_step[output_index] = agents_src->time_step[index];        
		agents_dst->lambda[output_index] = agents_src->lambda[index];        
		agents_dst->timevisiting[output_index] = agents_src->timevisiting[index];        
		agents_dst->bargoing[output_index] = agents_src->bargoing[index];        
		agents_dst->barday[output_index] = agents_src->barday[index];
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
	    agents_dst->householdtime[output_index] = agents_src->householdtime[index];
	    agents_dst->churchtime[output_index] = agents_src->churchtime[index];
	    agents_dst->transporttime[output_index] = agents_src->transporttime[index];
	    agents_dst->clinictime[output_index] = agents_src->clinictime[index];
	    agents_dst->workplacetime[output_index] = agents_src->workplacetime[index];
	    agents_dst->outsidetime[output_index] = agents_src->outsidetime[index];
	    agents_dst->age[output_index] = agents_src->age[index];
	    agents_dst->gender[output_index] = agents_src->gender[index];
	    agents_dst->householdsize[output_index] = agents_src->householdsize[index];
	    agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];
	    agents_dst->churchdur[output_index] = agents_src->churchdur[index];
	    agents_dst->transportdur[output_index] = agents_src->transportdur[index];
	    agents_dst->transportday1[output_index] = agents_src->transportday1[index];
	    agents_dst->transportday2[output_index] = agents_src->transportday2[index];
	    agents_dst->household[output_index] = agents_src->household[index];
	    agents_dst->church[output_index] = agents_src->church[index];
	    agents_dst->transport[output_index] = agents_src->transport[index];
	    agents_dst->workplace[output_index] = agents_src->workplace[index];
	    agents_dst->busy[output_index] = agents_src->busy[index];
	    agents_dst->startstep[output_index] = agents_src->startstep[index];
	    agents_dst->location[output_index] = agents_src->location[index];
	    agents_dst->locationid[output_index] = agents_src->locationid[index];
	    agents_dst->hiv[output_index] = agents_src->hiv[index];
	    agents_dst->art[output_index] = agents_src->art[index];
	    agents_dst->activetb[output_index] = agents_src->activetb[index];
	    agents_dst->artday[output_index] = agents_src->artday[index];
	    agents_dst->p[output_index] = agents_src->p[index];
	    agents_dst->q[output_index] = agents_src->q[index];
	    agents_dst->infections[output_index] = agents_src->infections[index];
	    agents_dst->lastinfected[output_index] = agents_src->lastinfected[index];
	    agents_dst->lastinfectedid[output_index] = agents_src->lastinfectedid[index];
	    agents_dst->time_step[output_index] = agents_src->time_step[index];
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
	    agents_dst->timevisiting[output_index] = agents_src->timevisiting[index];
	    agents_dst->bargoing[output_index] = agents_src->bargoing[index];
	    agents_dst->barday[output_index] = agents_src->barday[index];
    }
}

/** add_Person_agent
 * Continuous Person agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Person_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param step agent variable of type unsigned int
 * @param householdtime agent variable of type unsigned int
 * @param churchtime agent variable of type unsigned int
 * @param transporttime agent variable of type unsigned int
 * @param clinictime agent variable of type unsigned int
 * @param workplacetime agent variable of type unsigned int
 * @param outsidetime agent variable of type unsigned int
 * @param age agent variable of type unsigned int
 * @param gender agent variable of type unsigned int
 * @param householdsize agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 * @param churchdur agent variable of type float
 * @param transportdur agent variable of type unsigned int
 * @param transportday1 agent variable of type int
 * @param transportday2 agent variable of type int
 * @param household agent variable of type unsigned int
 * @param church agent variable of type int
 * @param transport agent variable of type int
 * @param workplace agent variable of type int
 * @param busy agent variable of type unsigned int
 * @param startstep agent variable of type unsigned int
 * @param location agent variable of type unsigned int
 * @param locationid agent variable of type unsigned int
 * @param hiv agent variable of type unsigned int
 * @param art agent variable of type unsigned int
 * @param activetb agent variable of type unsigned int
 * @param artday agent variable of type unsigned int
 * @param p agent variable of type float
 * @param q agent variable of type float
 * @param infections agent variable of type unsigned int
 * @param lastinfected agent variable of type int
 * @param lastinfectedid agent variable of type int
 * @param time_step agent variable of type float
 * @param lambda agent variable of type float
 * @param timevisiting agent variable of type unsigned int
 * @param bargoing agent variable of type unsigned int
 * @param barday agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int step, unsigned int householdtime, unsigned int churchtime, unsigned int transporttime, unsigned int clinictime, unsigned int workplacetime, unsigned int outsidetime, unsigned int age, unsigned int gender, unsigned int householdsize, unsigned int churchfreq, float churchdur, unsigned int transportdur, int transportday1, int transportday2, unsigned int household, int church, int transport, int workplace, unsigned int busy, unsigned int startstep, unsigned int location, unsigned int locationid, unsigned int hiv, unsigned int art, unsigned int activetb, unsigned int artday, float p, float q, unsigned int infections, int lastinfected, int lastinfectedid, float time_step, float lambda, unsigned int timevisiting, unsigned int bargoing, unsigned int barday){
	
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
	agents->householdtime[index] = householdtime;
	agents->churchtime[index] = churchtime;
	agents->transporttime[index] = transporttime;
	agents->clinictime[index] = clinictime;
	agents->workplacetime[index] = workplacetime;
	agents->outsidetime[index] = outsidetime;
	agents->age[index] = age;
	agents->gender[index] = gender;
	agents->householdsize[index] = householdsize;
	agents->churchfreq[index] = churchfreq;
	agents->churchdur[index] = churchdur;
	agents->transportdur[index] = transportdur;
	agents->transportday1[index] = transportday1;
	agents->transportday2[index] = transportday2;
	agents->household[index] = household;
	agents->church[index] = church;
	agents->transport[index] = transport;
	agents->workplace[index] = workplace;
	agents->busy[index] = busy;
	agents->startstep[index] = startstep;
	agents->location[index] = location;
	agents->locationid[index] = locationid;
	agents->hiv[index] = hiv;
	agents->art[index] = art;
	agents->activetb[index] = activetb;
	agents->artday[index] = artday;
	agents->p[index] = p;
	agents->q[index] = q;
	agents->infections[index] = infections;
	agents->lastinfected[index] = lastinfected;
	agents->lastinfectedid[index] = lastinfectedid;
	agents->time_step[index] = time_step;
	agents->lambda[index] = lambda;
	agents->timevisiting[index] = timevisiting;
	agents->bargoing[index] = bargoing;
	agents->barday[index] = barday;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int step, unsigned int householdtime, unsigned int churchtime, unsigned int transporttime, unsigned int clinictime, unsigned int workplacetime, unsigned int outsidetime, unsigned int age, unsigned int gender, unsigned int householdsize, unsigned int churchfreq, float churchdur, unsigned int transportdur, int transportday1, int transportday2, unsigned int household, int church, int transport, int workplace, unsigned int busy, unsigned int startstep, unsigned int location, unsigned int locationid, unsigned int hiv, unsigned int art, unsigned int activetb, unsigned int artday, float p, float q, unsigned int infections, int lastinfected, int lastinfectedid, float time_step, float lambda, unsigned int timevisiting, unsigned int bargoing, unsigned int barday){
    add_Person_agent<DISCRETE_2D>(agents, id, step, householdtime, churchtime, transporttime, clinictime, workplacetime, outsidetime, age, gender, householdsize, churchfreq, churchdur, transportdur, transportday1, transportday2, household, church, transport, workplace, busy, startstep, location, locationid, hiv, art, activetb, artday, p, q, infections, lastinfected, lastinfectedid, time_step, lambda, timevisiting, bargoing, barday);
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
	ordered_agents->householdtime[index] = unordered_agents->householdtime[old_pos];
	ordered_agents->churchtime[index] = unordered_agents->churchtime[old_pos];
	ordered_agents->transporttime[index] = unordered_agents->transporttime[old_pos];
	ordered_agents->clinictime[index] = unordered_agents->clinictime[old_pos];
	ordered_agents->workplacetime[index] = unordered_agents->workplacetime[old_pos];
	ordered_agents->outsidetime[index] = unordered_agents->outsidetime[old_pos];
	ordered_agents->age[index] = unordered_agents->age[old_pos];
	ordered_agents->gender[index] = unordered_agents->gender[old_pos];
	ordered_agents->householdsize[index] = unordered_agents->householdsize[old_pos];
	ordered_agents->churchfreq[index] = unordered_agents->churchfreq[old_pos];
	ordered_agents->churchdur[index] = unordered_agents->churchdur[old_pos];
	ordered_agents->transportdur[index] = unordered_agents->transportdur[old_pos];
	ordered_agents->transportday1[index] = unordered_agents->transportday1[old_pos];
	ordered_agents->transportday2[index] = unordered_agents->transportday2[old_pos];
	ordered_agents->household[index] = unordered_agents->household[old_pos];
	ordered_agents->church[index] = unordered_agents->church[old_pos];
	ordered_agents->transport[index] = unordered_agents->transport[old_pos];
	ordered_agents->workplace[index] = unordered_agents->workplace[old_pos];
	ordered_agents->busy[index] = unordered_agents->busy[old_pos];
	ordered_agents->startstep[index] = unordered_agents->startstep[old_pos];
	ordered_agents->location[index] = unordered_agents->location[old_pos];
	ordered_agents->locationid[index] = unordered_agents->locationid[old_pos];
	ordered_agents->hiv[index] = unordered_agents->hiv[old_pos];
	ordered_agents->art[index] = unordered_agents->art[old_pos];
	ordered_agents->activetb[index] = unordered_agents->activetb[old_pos];
	ordered_agents->artday[index] = unordered_agents->artday[old_pos];
	ordered_agents->p[index] = unordered_agents->p[old_pos];
	ordered_agents->q[index] = unordered_agents->q[old_pos];
	ordered_agents->infections[index] = unordered_agents->infections[old_pos];
	ordered_agents->lastinfected[index] = unordered_agents->lastinfected[old_pos];
	ordered_agents->lastinfectedid[index] = unordered_agents->lastinfectedid[old_pos];
	ordered_agents->time_step[index] = unordered_agents->time_step[old_pos];
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
	ordered_agents->timevisiting[index] = unordered_agents->timevisiting[old_pos];
	ordered_agents->bargoing[index] = unordered_agents->bargoing[old_pos];
	ordered_agents->barday[index] = unordered_agents->barday[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created TBAssignment agent functions */

/** reset_TBAssignment_scan_input
 * TBAssignment agent reset scan input function
 * @param agents The xmachine_memory_TBAssignment_list agent list
 */
__global__ void reset_TBAssignment_scan_input(xmachine_memory_TBAssignment_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_TBAssignment_Agents
 * TBAssignment scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TBAssignment_list agent list destination
 * @param agents_src xmachine_memory_TBAssignment_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_TBAssignment_Agents(xmachine_memory_TBAssignment_list* agents_dst, xmachine_memory_TBAssignment_list* agents_src, int dst_agent_count, int number_to_scatter){
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
	}
}

/** append_TBAssignment_Agents
 * TBAssignment scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TBAssignment_list agent list destination
 * @param agents_src xmachine_memory_TBAssignment_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_TBAssignment_Agents(xmachine_memory_TBAssignment_list* agents_dst, xmachine_memory_TBAssignment_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
    }
}

/** add_TBAssignment_agent
 * Continuous TBAssignment agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_TBAssignment_list to add agents to 
 * @param id agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_TBAssignment_agent(xmachine_memory_TBAssignment_list* agents, unsigned int id){
	
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

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_TBAssignment_agent(xmachine_memory_TBAssignment_list* agents, unsigned int id){
    add_TBAssignment_agent<DISCRETE_2D>(agents, id);
}

/** reorder_TBAssignment_agents
 * Continuous TBAssignment agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_TBAssignment_agents(unsigned int* values, xmachine_memory_TBAssignment_list* unordered_agents, xmachine_memory_TBAssignment_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];        
		agents_dst->active[output_index] = agents_src->active[index];
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
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
	    agents_dst->active[output_index] = agents_src->active[index];
    }
}

/** add_Household_agent
 * Continuous Household agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Household_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param lambda agent variable of type float
 * @param active agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, float lambda, unsigned int active){
	
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
	agents->lambda[index] = lambda;
	agents->active[index] = active;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, float lambda, unsigned int active){
    add_Household_agent<DISCRETE_2D>(agents, id, lambda, active);
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
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
	ordered_agents->active[index] = unordered_agents->active[old_pos];
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
		agents_dst->household_size[output_index] = agents_src->household_size[index];        
		agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];        
		agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];
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
	    agents_dst->household_size[output_index] = agents_src->household_size[index];
	    agents_dst->churchgoing[output_index] = agents_src->churchgoing[index];
	    agents_dst->churchfreq[output_index] = agents_src->churchfreq[index];
    }
}

/** add_HouseholdMembership_agent
 * Continuous HouseholdMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_HouseholdMembership_list to add agents to 
 * @param household_id agent variable of type unsigned int
 * @param person_id agent variable of type unsigned int
 * @param household_size agent variable of type unsigned int
 * @param churchgoing agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_HouseholdMembership_agent(xmachine_memory_HouseholdMembership_list* agents, unsigned int household_id, unsigned int person_id, unsigned int household_size, unsigned int churchgoing, unsigned int churchfreq){
	
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
	agents->household_size[index] = household_size;
	agents->churchgoing[index] = churchgoing;
	agents->churchfreq[index] = churchfreq;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_HouseholdMembership_agent(xmachine_memory_HouseholdMembership_list* agents, unsigned int household_id, unsigned int person_id, unsigned int household_size, unsigned int churchgoing, unsigned int churchfreq){
    add_HouseholdMembership_agent<DISCRETE_2D>(agents, household_id, person_id, household_size, churchgoing, churchfreq);
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
	ordered_agents->household_size[index] = unordered_agents->household_size[old_pos];
	ordered_agents->churchgoing[index] = unordered_agents->churchgoing[old_pos];
	ordered_agents->churchfreq[index] = unordered_agents->churchfreq[old_pos];
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];        
		agents_dst->active[output_index] = agents_src->active[index];
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
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
	    agents_dst->active[output_index] = agents_src->active[index];
    }
}

/** add_Church_agent
 * Continuous Church agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Church_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param lambda agent variable of type float
 * @param active agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float lambda, unsigned int active){
	
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
	agents->lambda[index] = lambda;
	agents->active[index] = active;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float lambda, unsigned int active){
    add_Church_agent<DISCRETE_2D>(agents, id, size, lambda, active);
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
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
	ordered_agents->active[index] = unordered_agents->active[old_pos];
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
		agents_dst->churchdur[output_index] = agents_src->churchdur[index];
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
	    agents_dst->churchdur[output_index] = agents_src->churchdur[index];
    }
}

/** add_ChurchMembership_agent
 * Continuous ChurchMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_ChurchMembership_list to add agents to 
 * @param church_id agent variable of type unsigned int
 * @param household_id agent variable of type unsigned int
 * @param churchdur agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_ChurchMembership_agent(xmachine_memory_ChurchMembership_list* agents, unsigned int church_id, unsigned int household_id, float churchdur){
	
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
	agents->churchdur[index] = churchdur;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_ChurchMembership_agent(xmachine_memory_ChurchMembership_list* agents, unsigned int church_id, unsigned int household_id, float churchdur){
    add_ChurchMembership_agent<DISCRETE_2D>(agents, church_id, household_id, churchdur);
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
	ordered_agents->churchdur[index] = unordered_agents->churchdur[old_pos];
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];        
		agents_dst->active[output_index] = agents_src->active[index];
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
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
	    agents_dst->active[output_index] = agents_src->active[index];
    }
}

/** add_Transport_agent
 * Continuous Transport agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Transport_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param lambda agent variable of type float
 * @param active agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_Transport_agent(xmachine_memory_Transport_list* agents, unsigned int id, float lambda, unsigned int active){
	
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
	agents->lambda[index] = lambda;
	agents->active[index] = active;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Transport_agent(xmachine_memory_Transport_list* agents, unsigned int id, float lambda, unsigned int active){
    add_Transport_agent<DISCRETE_2D>(agents, id, lambda, active);
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
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
	ordered_agents->active[index] = unordered_agents->active[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created TransportMembership agent functions */

/** reset_TransportMembership_scan_input
 * TransportMembership agent reset scan input function
 * @param agents The xmachine_memory_TransportMembership_list agent list
 */
__global__ void reset_TransportMembership_scan_input(xmachine_memory_TransportMembership_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_TransportMembership_Agents
 * TransportMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TransportMembership_list agent list destination
 * @param agents_src xmachine_memory_TransportMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_TransportMembership_Agents(xmachine_memory_TransportMembership_list* agents_dst, xmachine_memory_TransportMembership_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->person_id[output_index] = agents_src->person_id[index];        
		agents_dst->transport_id[output_index] = agents_src->transport_id[index];        
		agents_dst->duration[output_index] = agents_src->duration[index];
	}
}

/** append_TransportMembership_Agents
 * TransportMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_TransportMembership_list agent list destination
 * @param agents_src xmachine_memory_TransportMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_TransportMembership_Agents(xmachine_memory_TransportMembership_list* agents_dst, xmachine_memory_TransportMembership_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->person_id[output_index] = agents_src->person_id[index];
	    agents_dst->transport_id[output_index] = agents_src->transport_id[index];
	    agents_dst->duration[output_index] = agents_src->duration[index];
    }
}

/** add_TransportMembership_agent
 * Continuous TransportMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_TransportMembership_list to add agents to 
 * @param person_id agent variable of type int
 * @param transport_id agent variable of type unsigned int
 * @param duration agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_TransportMembership_agent(xmachine_memory_TransportMembership_list* agents, int person_id, unsigned int transport_id, unsigned int duration){
	
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
	agents->person_id[index] = person_id;
	agents->transport_id[index] = transport_id;
	agents->duration[index] = duration;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_TransportMembership_agent(xmachine_memory_TransportMembership_list* agents, int person_id, unsigned int transport_id, unsigned int duration){
    add_TransportMembership_agent<DISCRETE_2D>(agents, person_id, transport_id, duration);
}

/** reorder_TransportMembership_agents
 * Continuous TransportMembership agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_TransportMembership_agents(unsigned int* values, xmachine_memory_TransportMembership_list* unordered_agents, xmachine_memory_TransportMembership_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->person_id[index] = unordered_agents->person_id[old_pos];
	ordered_agents->transport_id[index] = unordered_agents->transport_id[old_pos];
	ordered_agents->duration[index] = unordered_agents->duration[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Clinic agent functions */

/** reset_Clinic_scan_input
 * Clinic agent reset scan input function
 * @param agents The xmachine_memory_Clinic_list agent list
 */
__global__ void reset_Clinic_scan_input(xmachine_memory_Clinic_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Clinic_Agents
 * Clinic scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Clinic_list agent list destination
 * @param agents_src xmachine_memory_Clinic_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Clinic_Agents(xmachine_memory_Clinic_list* agents_dst, xmachine_memory_Clinic_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];
	}
}

/** append_Clinic_Agents
 * Clinic scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Clinic_list agent list destination
 * @param agents_src xmachine_memory_Clinic_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Clinic_Agents(xmachine_memory_Clinic_list* agents_dst, xmachine_memory_Clinic_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
    }
}

/** add_Clinic_agent
 * Continuous Clinic agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Clinic_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param lambda agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Clinic_agent(xmachine_memory_Clinic_list* agents, unsigned int id, float lambda){
	
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
	agents->lambda[index] = lambda;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Clinic_agent(xmachine_memory_Clinic_list* agents, unsigned int id, float lambda){
    add_Clinic_agent<DISCRETE_2D>(agents, id, lambda);
}

/** reorder_Clinic_agents
 * Continuous Clinic agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Clinic_agents(unsigned int* values, xmachine_memory_Clinic_list* unordered_agents, xmachine_memory_Clinic_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Workplace agent functions */

/** reset_Workplace_scan_input
 * Workplace agent reset scan input function
 * @param agents The xmachine_memory_Workplace_list agent list
 */
__global__ void reset_Workplace_scan_input(xmachine_memory_Workplace_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Workplace_Agents
 * Workplace scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Workplace_list agent list destination
 * @param agents_src xmachine_memory_Workplace_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Workplace_Agents(xmachine_memory_Workplace_list* agents_dst, xmachine_memory_Workplace_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];
	}
}

/** append_Workplace_Agents
 * Workplace scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Workplace_list agent list destination
 * @param agents_src xmachine_memory_Workplace_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Workplace_Agents(xmachine_memory_Workplace_list* agents_dst, xmachine_memory_Workplace_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
    }
}

/** add_Workplace_agent
 * Continuous Workplace agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Workplace_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param lambda agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Workplace_agent(xmachine_memory_Workplace_list* agents, unsigned int id, float lambda){
	
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
	agents->lambda[index] = lambda;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Workplace_agent(xmachine_memory_Workplace_list* agents, unsigned int id, float lambda){
    add_Workplace_agent<DISCRETE_2D>(agents, id, lambda);
}

/** reorder_Workplace_agents
 * Continuous Workplace agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Workplace_agents(unsigned int* values, xmachine_memory_Workplace_list* unordered_agents, xmachine_memory_Workplace_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created WorkplaceMembership agent functions */

/** reset_WorkplaceMembership_scan_input
 * WorkplaceMembership agent reset scan input function
 * @param agents The xmachine_memory_WorkplaceMembership_list agent list
 */
__global__ void reset_WorkplaceMembership_scan_input(xmachine_memory_WorkplaceMembership_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_WorkplaceMembership_Agents
 * WorkplaceMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_WorkplaceMembership_list agent list destination
 * @param agents_src xmachine_memory_WorkplaceMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_WorkplaceMembership_Agents(xmachine_memory_WorkplaceMembership_list* agents_dst, xmachine_memory_WorkplaceMembership_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->person_id[output_index] = agents_src->person_id[index];        
		agents_dst->workplace_id[output_index] = agents_src->workplace_id[index];
	}
}

/** append_WorkplaceMembership_Agents
 * WorkplaceMembership scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_WorkplaceMembership_list agent list destination
 * @param agents_src xmachine_memory_WorkplaceMembership_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_WorkplaceMembership_Agents(xmachine_memory_WorkplaceMembership_list* agents_dst, xmachine_memory_WorkplaceMembership_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->person_id[output_index] = agents_src->person_id[index];
	    agents_dst->workplace_id[output_index] = agents_src->workplace_id[index];
    }
}

/** add_WorkplaceMembership_agent
 * Continuous WorkplaceMembership agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_WorkplaceMembership_list to add agents to 
 * @param person_id agent variable of type unsigned int
 * @param workplace_id agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_WorkplaceMembership_agent(xmachine_memory_WorkplaceMembership_list* agents, unsigned int person_id, unsigned int workplace_id){
	
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
	agents->person_id[index] = person_id;
	agents->workplace_id[index] = workplace_id;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_WorkplaceMembership_agent(xmachine_memory_WorkplaceMembership_list* agents, unsigned int person_id, unsigned int workplace_id){
    add_WorkplaceMembership_agent<DISCRETE_2D>(agents, person_id, workplace_id);
}

/** reorder_WorkplaceMembership_agents
 * Continuous WorkplaceMembership agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_WorkplaceMembership_agents(unsigned int* values, xmachine_memory_WorkplaceMembership_list* unordered_agents, xmachine_memory_WorkplaceMembership_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->person_id[index] = unordered_agents->person_id[old_pos];
	ordered_agents->workplace_id[index] = unordered_agents->workplace_id[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Bar agent functions */

/** reset_Bar_scan_input
 * Bar agent reset scan input function
 * @param agents The xmachine_memory_Bar_list agent list
 */
__global__ void reset_Bar_scan_input(xmachine_memory_Bar_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Bar_Agents
 * Bar scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Bar_list agent list destination
 * @param agents_src xmachine_memory_Bar_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Bar_Agents(xmachine_memory_Bar_list* agents_dst, xmachine_memory_Bar_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->lambda[output_index] = agents_src->lambda[index];
	}
}

/** append_Bar_Agents
 * Bar scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Bar_list agent list destination
 * @param agents_src xmachine_memory_Bar_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Bar_Agents(xmachine_memory_Bar_list* agents_dst, xmachine_memory_Bar_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->lambda[output_index] = agents_src->lambda[index];
    }
}

/** add_Bar_agent
 * Continuous Bar agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Bar_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param lambda agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Bar_agent(xmachine_memory_Bar_list* agents, unsigned int id, float lambda){
	
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
	agents->lambda[index] = lambda;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Bar_agent(xmachine_memory_Bar_list* agents, unsigned int id, float lambda){
    add_Bar_agent<DISCRETE_2D>(agents, id, lambda);
}

/** reorder_Bar_agents
 * Continuous Bar agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Bar_agents(unsigned int* values, xmachine_memory_Bar_list* unordered_agents, xmachine_memory_Bar_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->lambda[index] = unordered_agents->lambda[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created tb_assignment message functions */


/** add_tb_assignment_message
 * Add non partitioned or spatially partitioned tb_assignment message
 * @param messages xmachine_message_tb_assignment_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_tb_assignment_message(xmachine_message_tb_assignment_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_tb_assignment_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_tb_assignment_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_tb_assignment_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_tb_assignment Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned tb_assignment message (for optional messages)
 * @param messages scatter_optional_tb_assignment_messages Sparse xmachine_message_tb_assignment_list message list
 * @param message_swap temp xmachine_message_tb_assignment_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_tb_assignment_messages(xmachine_message_tb_assignment_list* messages, xmachine_message_tb_assignment_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_tb_assignment_count;

		//AoS - xmachine_message_tb_assignment Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_tb_assignment_swaps
 * Reset non partitioned or spatially partitioned tb_assignment message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_tb_assignment_swaps(xmachine_message_tb_assignment_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_tb_assignment* get_first_tb_assignment_message(xmachine_message_tb_assignment_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_tb_assignment_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_tb_assignment Coalesced memory read
	xmachine_message_tb_assignment temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_tb_assignment));
	xmachine_message_tb_assignment* sm_message = ((xmachine_message_tb_assignment*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_tb_assignment*)&message_share[d_SM_START]);
}

__device__ xmachine_message_tb_assignment* get_next_tb_assignment_message(xmachine_message_tb_assignment* message, xmachine_message_tb_assignment_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_tb_assignment_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_tb_assignment_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_tb_assignment Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_tb_assignment temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_tb_assignment));
		xmachine_message_tb_assignment* sm_message = ((xmachine_message_tb_assignment*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_tb_assignment));
	return ((xmachine_message_tb_assignment*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created household_membership message functions */


/** add_household_membership_message
 * Add non partitioned or spatially partitioned household_membership message
 * @param messages xmachine_message_household_membership_list message list to add too
 * @param household_id agent variable of type unsigned int
 * @param person_id agent variable of type unsigned int
 * @param household_size agent variable of type unsigned int
 * @param church_id agent variable of type unsigned int
 * @param churchfreq agent variable of type unsigned int
 * @param churchdur agent variable of type float
 */
__device__ void add_household_membership_message(xmachine_message_household_membership_list* messages, unsigned int household_id, unsigned int person_id, unsigned int household_size, unsigned int church_id, unsigned int churchfreq, float churchdur){

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
	messages->household_size[index] = household_size;
	messages->church_id[index] = church_id;
	messages->churchfreq[index] = churchfreq;
	messages->churchdur[index] = churchdur;

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
		messages->household_size[output_index] = messages_swap->household_size[index];
		messages->church_id[output_index] = messages_swap->church_id[index];
		messages->churchfreq[output_index] = messages_swap->churchfreq[index];
		messages->churchdur[output_index] = messages_swap->churchdur[index];				
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
	temp_message.household_size = messages->household_size[index];
	temp_message.church_id = messages->church_id[index];
	temp_message.churchfreq = messages->churchfreq[index];
	temp_message.churchdur = messages->churchdur[index];

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
		temp_message.household_size = messages->household_size[index];
		temp_message.church_id = messages->church_id[index];
		temp_message.churchfreq = messages->churchfreq[index];
		temp_message.churchdur = messages->churchdur[index];

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
 * @param churchdur agent variable of type float
 */
__device__ void add_church_membership_message(xmachine_message_church_membership_list* messages, unsigned int church_id, unsigned int household_id, float churchdur){

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
	messages->churchdur[index] = churchdur;

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
		messages->churchdur[output_index] = messages_swap->churchdur[index];				
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
	temp_message.churchdur = messages->churchdur[index];

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
		temp_message.churchdur = messages->churchdur[index];

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
/* Dyanamically created transport_membership message functions */


/** add_transport_membership_message
 * Add non partitioned or spatially partitioned transport_membership message
 * @param messages xmachine_message_transport_membership_list message list to add too
 * @param person_id agent variable of type unsigned int
 * @param transport_id agent variable of type unsigned int
 * @param duration agent variable of type unsigned int
 */
__device__ void add_transport_membership_message(xmachine_message_transport_membership_list* messages, unsigned int person_id, unsigned int transport_id, unsigned int duration){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_transport_membership_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_transport_membership_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_transport_membership_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_transport_membership Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->person_id[index] = person_id;
	messages->transport_id[index] = transport_id;
	messages->duration[index] = duration;

}

/**
 * Scatter non partitioned or spatially partitioned transport_membership message (for optional messages)
 * @param messages scatter_optional_transport_membership_messages Sparse xmachine_message_transport_membership_list message list
 * @param message_swap temp xmachine_message_transport_membership_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_transport_membership_messages(xmachine_message_transport_membership_list* messages, xmachine_message_transport_membership_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_transport_membership_count;

		//AoS - xmachine_message_transport_membership Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->person_id[output_index] = messages_swap->person_id[index];
		messages->transport_id[output_index] = messages_swap->transport_id[index];
		messages->duration[output_index] = messages_swap->duration[index];				
	}
}

/** reset_transport_membership_swaps
 * Reset non partitioned or spatially partitioned transport_membership message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_transport_membership_swaps(xmachine_message_transport_membership_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_transport_membership* get_first_transport_membership_message(xmachine_message_transport_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_transport_membership_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_transport_membership Coalesced memory read
	xmachine_message_transport_membership temp_message;
	temp_message._position = messages->_position[index];
	temp_message.person_id = messages->person_id[index];
	temp_message.transport_id = messages->transport_id[index];
	temp_message.duration = messages->duration[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_transport_membership));
	xmachine_message_transport_membership* sm_message = ((xmachine_message_transport_membership*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_transport_membership*)&message_share[d_SM_START]);
}

__device__ xmachine_message_transport_membership* get_next_transport_membership_message(xmachine_message_transport_membership* message, xmachine_message_transport_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_transport_membership_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_transport_membership_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_transport_membership Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_transport_membership temp_message;
		temp_message._position = messages->_position[index];
		temp_message.person_id = messages->person_id[index];
		temp_message.transport_id = messages->transport_id[index];
		temp_message.duration = messages->duration[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_transport_membership));
		xmachine_message_transport_membership* sm_message = ((xmachine_message_transport_membership*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_transport_membership));
	return ((xmachine_message_transport_membership*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created workplace_membership message functions */


/** add_workplace_membership_message
 * Add non partitioned or spatially partitioned workplace_membership message
 * @param messages xmachine_message_workplace_membership_list message list to add too
 * @param person_id agent variable of type unsigned int
 * @param workplace_id agent variable of type unsigned int
 */
__device__ void add_workplace_membership_message(xmachine_message_workplace_membership_list* messages, unsigned int person_id, unsigned int workplace_id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_workplace_membership_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_workplace_membership_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_workplace_membership_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_workplace_membership Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->person_id[index] = person_id;
	messages->workplace_id[index] = workplace_id;

}

/**
 * Scatter non partitioned or spatially partitioned workplace_membership message (for optional messages)
 * @param messages scatter_optional_workplace_membership_messages Sparse xmachine_message_workplace_membership_list message list
 * @param message_swap temp xmachine_message_workplace_membership_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_workplace_membership_messages(xmachine_message_workplace_membership_list* messages, xmachine_message_workplace_membership_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_workplace_membership_count;

		//AoS - xmachine_message_workplace_membership Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->person_id[output_index] = messages_swap->person_id[index];
		messages->workplace_id[output_index] = messages_swap->workplace_id[index];				
	}
}

/** reset_workplace_membership_swaps
 * Reset non partitioned or spatially partitioned workplace_membership message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_workplace_membership_swaps(xmachine_message_workplace_membership_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_workplace_membership* get_first_workplace_membership_message(xmachine_message_workplace_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_workplace_membership_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_workplace_membership Coalesced memory read
	xmachine_message_workplace_membership temp_message;
	temp_message._position = messages->_position[index];
	temp_message.person_id = messages->person_id[index];
	temp_message.workplace_id = messages->workplace_id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_workplace_membership));
	xmachine_message_workplace_membership* sm_message = ((xmachine_message_workplace_membership*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_workplace_membership*)&message_share[d_SM_START]);
}

__device__ xmachine_message_workplace_membership* get_next_workplace_membership_message(xmachine_message_workplace_membership* message, xmachine_message_workplace_membership_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_workplace_membership_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_workplace_membership_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_workplace_membership Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_workplace_membership temp_message;
		temp_message._position = messages->_position[index];
		temp_message.person_id = messages->person_id[index];
		temp_message.workplace_id = messages->workplace_id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_workplace_membership));
		xmachine_message_workplace_membership* sm_message = ((xmachine_message_workplace_membership*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_workplace_membership));
	return ((xmachine_message_workplace_membership*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created location message functions */


/** add_location_message
 * Add non partitioned or spatially partitioned location message
 * @param messages xmachine_message_location_list message list to add too
 * @param person_id agent variable of type unsigned int
 * @param location agent variable of type unsigned int
 * @param locationid agent variable of type unsigned int
 * @param p agent variable of type float
 * @param q agent variable of type float
 */
__device__ void add_location_message(xmachine_message_location_list* messages, unsigned int person_id, unsigned int location, unsigned int locationid, float p, float q){

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
	messages->location[index] = location;
	messages->locationid[index] = locationid;
	messages->p[index] = p;
	messages->q[index] = q;

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
		messages->location[output_index] = messages_swap->location[index];
		messages->locationid[output_index] = messages_swap->locationid[index];
		messages->p[output_index] = messages_swap->p[index];
		messages->q[output_index] = messages_swap->q[index];				
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
	temp_message.location = messages->location[index];
	temp_message.locationid = messages->locationid[index];
	temp_message.p = messages->p[index];
	temp_message.q = messages->q[index];

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
		temp_message.location = messages->location[index];
		temp_message.locationid = messages->locationid[index];
		temp_message.p = messages->p[index];
		temp_message.q = messages->q[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_location));
		xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_location));
	return ((xmachine_message_location*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created infection message functions */


/** add_infection_message
 * Add non partitioned or spatially partitioned infection message
 * @param messages xmachine_message_infection_list message list to add too
 * @param location agent variable of type unsigned int
 * @param locationid agent variable of type unsigned int
 * @param lambda agent variable of type float
 */
__device__ void add_infection_message(xmachine_message_infection_list* messages, unsigned int location, unsigned int locationid, float lambda){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_infection_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_infection_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_infection_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_infection Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->location[index] = location;
	messages->locationid[index] = locationid;
	messages->lambda[index] = lambda;

}

/**
 * Scatter non partitioned or spatially partitioned infection message (for optional messages)
 * @param messages scatter_optional_infection_messages Sparse xmachine_message_infection_list message list
 * @param message_swap temp xmachine_message_infection_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_infection_messages(xmachine_message_infection_list* messages, xmachine_message_infection_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_infection_count;

		//AoS - xmachine_message_infection Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->location[output_index] = messages_swap->location[index];
		messages->locationid[output_index] = messages_swap->locationid[index];
		messages->lambda[output_index] = messages_swap->lambda[index];				
	}
}

/** reset_infection_swaps
 * Reset non partitioned or spatially partitioned infection message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_infection_swaps(xmachine_message_infection_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_infection* get_first_infection_message(xmachine_message_infection_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_infection_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_infection Coalesced memory read
	xmachine_message_infection temp_message;
	temp_message._position = messages->_position[index];
	temp_message.location = messages->location[index];
	temp_message.locationid = messages->locationid[index];
	temp_message.lambda = messages->lambda[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_infection));
	xmachine_message_infection* sm_message = ((xmachine_message_infection*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_infection*)&message_share[d_SM_START]);
}

__device__ xmachine_message_infection* get_next_infection_message(xmachine_message_infection* message, xmachine_message_infection_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_infection_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_infection_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_infection Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_infection temp_message;
		temp_message._position = messages->_position[index];
		temp_message.location = messages->location[index];
		temp_message.locationid = messages->locationid[index];
		temp_message.lambda = messages->lambda[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_infection));
		xmachine_message_infection* sm_message = ((xmachine_message_infection*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_infection));
	return ((xmachine_message_infection*)&message_share[message_index]);
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_update(xmachine_memory_Person_list* agents, xmachine_message_location_list* location_messages, RNG_rand48* rand48){
	
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
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];

	//FLAME function call
	int dead = !update(&agent, location_messages	, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_update Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
}

/**
 *
 */
__global__ void GPUFLAME_updatelambda(xmachine_memory_Person_list* agents, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_updatelambda Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.householdtime = 0;
	agent.churchtime = 0;
	agent.transporttime = 0;
	agent.clinictime = 0;
	agent.workplacetime = 0;
	agent.outsidetime = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportdur = 0;
	agent.transportday1 = 0;
	agent.transportday2 = 0;
	agent.household = 0;
	agent.church = 0;
	agent.transport = 0;
	agent.workplace = 0;
	agent.busy = 0;
	agent.startstep = 0;
	agent.location = 0;
	agent.locationid = 0;
	agent.hiv = 0;
	agent.art = 0;
	agent.activetb = 0;
	agent.artday = 0;
	agent.p = 0;
	agent.q = 0;
	agent.infections = 0;
	agent.lastinfected = 0;
	agent.lastinfectedid = 0;
	agent.time_step = 0;
	agent.lambda = 0;
	agent.timevisiting = 0;
	agent.bargoing = 0;
	agent.barday = 0;
	}

	//FLAME function call
	int dead = !updatelambda(&agent, infection_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_updatelambda Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
	}
}

/**
 *
 */
__global__ void GPUFLAME_infect(xmachine_memory_Person_list* agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Person_count)
        return;
    

	//SoA to AoS - xmachine_memory_infect Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];

	//FLAME function call
	int dead = !infect(&agent, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_infect Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
}

/**
 *
 */
__global__ void GPUFLAME_personhhinit(xmachine_memory_Person_list* agents, xmachine_message_household_membership_list* household_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_personhhinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.householdtime = 0;
	agent.churchtime = 0;
	agent.transporttime = 0;
	agent.clinictime = 0;
	agent.workplacetime = 0;
	agent.outsidetime = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportdur = 0;
	agent.transportday1 = 0;
	agent.transportday2 = 0;
	agent.household = 0;
	agent.church = 0;
	agent.transport = 0;
	agent.workplace = 0;
	agent.busy = 0;
	agent.startstep = 0;
	agent.location = 0;
	agent.locationid = 0;
	agent.hiv = 0;
	agent.art = 0;
	agent.activetb = 0;
	agent.artday = 0;
	agent.p = 0;
	agent.q = 0;
	agent.infections = 0;
	agent.lastinfected = 0;
	agent.lastinfectedid = 0;
	agent.time_step = 0;
	agent.lambda = 0;
	agent.timevisiting = 0;
	agent.bargoing = 0;
	agent.barday = 0;
	}

	//FLAME function call
	int dead = !personhhinit(&agent, household_membership_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_personhhinit Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
	}
}

/**
 *
 */
__global__ void GPUFLAME_persontbinit(xmachine_memory_Person_list* agents, xmachine_message_tb_assignment_list* tb_assignment_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_persontbinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.householdtime = 0;
	agent.churchtime = 0;
	agent.transporttime = 0;
	agent.clinictime = 0;
	agent.workplacetime = 0;
	agent.outsidetime = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportdur = 0;
	agent.transportday1 = 0;
	agent.transportday2 = 0;
	agent.household = 0;
	agent.church = 0;
	agent.transport = 0;
	agent.workplace = 0;
	agent.busy = 0;
	agent.startstep = 0;
	agent.location = 0;
	agent.locationid = 0;
	agent.hiv = 0;
	agent.art = 0;
	agent.activetb = 0;
	agent.artday = 0;
	agent.p = 0;
	agent.q = 0;
	agent.infections = 0;
	agent.lastinfected = 0;
	agent.lastinfectedid = 0;
	agent.time_step = 0;
	agent.lambda = 0;
	agent.timevisiting = 0;
	agent.bargoing = 0;
	agent.barday = 0;
	}

	//FLAME function call
	int dead = !persontbinit(&agent, tb_assignment_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_persontbinit Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
	}
}

/**
 *
 */
__global__ void GPUFLAME_persontrinit(xmachine_memory_Person_list* agents, xmachine_message_transport_membership_list* transport_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_persontrinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.householdtime = 0;
	agent.churchtime = 0;
	agent.transporttime = 0;
	agent.clinictime = 0;
	agent.workplacetime = 0;
	agent.outsidetime = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportdur = 0;
	agent.transportday1 = 0;
	agent.transportday2 = 0;
	agent.household = 0;
	agent.church = 0;
	agent.transport = 0;
	agent.workplace = 0;
	agent.busy = 0;
	agent.startstep = 0;
	agent.location = 0;
	agent.locationid = 0;
	agent.hiv = 0;
	agent.art = 0;
	agent.activetb = 0;
	agent.artday = 0;
	agent.p = 0;
	agent.q = 0;
	agent.infections = 0;
	agent.lastinfected = 0;
	agent.lastinfectedid = 0;
	agent.time_step = 0;
	agent.lambda = 0;
	agent.timevisiting = 0;
	agent.bargoing = 0;
	agent.barday = 0;
	}

	//FLAME function call
	int dead = !persontrinit(&agent, transport_membership_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_persontrinit Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
	}
}

/**
 *
 */
__global__ void GPUFLAME_personwpinit(xmachine_memory_Person_list* agents, xmachine_message_workplace_membership_list* workplace_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_personwpinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Person agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    
	agent.id = agents->id[index];
	agent.step = agents->step[index];
	agent.householdtime = agents->householdtime[index];
	agent.churchtime = agents->churchtime[index];
	agent.transporttime = agents->transporttime[index];
	agent.clinictime = agents->clinictime[index];
	agent.workplacetime = agents->workplacetime[index];
	agent.outsidetime = agents->outsidetime[index];
	agent.age = agents->age[index];
	agent.gender = agents->gender[index];
	agent.householdsize = agents->householdsize[index];
	agent.churchfreq = agents->churchfreq[index];
	agent.churchdur = agents->churchdur[index];
	agent.transportdur = agents->transportdur[index];
	agent.transportday1 = agents->transportday1[index];
	agent.transportday2 = agents->transportday2[index];
	agent.household = agents->household[index];
	agent.church = agents->church[index];
	agent.transport = agents->transport[index];
	agent.workplace = agents->workplace[index];
	agent.busy = agents->busy[index];
	agent.startstep = agents->startstep[index];
	agent.location = agents->location[index];
	agent.locationid = agents->locationid[index];
	agent.hiv = agents->hiv[index];
	agent.art = agents->art[index];
	agent.activetb = agents->activetb[index];
	agent.artday = agents->artday[index];
	agent.p = agents->p[index];
	agent.q = agents->q[index];
	agent.infections = agents->infections[index];
	agent.lastinfected = agents->lastinfected[index];
	agent.lastinfectedid = agents->lastinfectedid[index];
	agent.time_step = agents->time_step[index];
	agent.lambda = agents->lambda[index];
	agent.timevisiting = agents->timevisiting[index];
	agent.bargoing = agents->bargoing[index];
	agent.barday = agents->barday[index];
	} else {
	
	agent.id = 0;
	agent.step = 0;
	agent.householdtime = 0;
	agent.churchtime = 0;
	agent.transporttime = 0;
	agent.clinictime = 0;
	agent.workplacetime = 0;
	agent.outsidetime = 0;
	agent.age = 0;
	agent.gender = 0;
	agent.householdsize = 0;
	agent.churchfreq = 0;
	agent.churchdur = 0;
	agent.transportdur = 0;
	agent.transportday1 = 0;
	agent.transportday2 = 0;
	agent.household = 0;
	agent.church = 0;
	agent.transport = 0;
	agent.workplace = 0;
	agent.busy = 0;
	agent.startstep = 0;
	agent.location = 0;
	agent.locationid = 0;
	agent.hiv = 0;
	agent.art = 0;
	agent.activetb = 0;
	agent.artday = 0;
	agent.p = 0;
	agent.q = 0;
	agent.infections = 0;
	agent.lastinfected = 0;
	agent.lastinfectedid = 0;
	agent.time_step = 0;
	agent.lambda = 0;
	agent.timevisiting = 0;
	agent.bargoing = 0;
	agent.barday = 0;
	}

	//FLAME function call
	int dead = !personwpinit(&agent, workplace_membership_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Person_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_personwpinit Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->step[index] = agent.step;
	agents->householdtime[index] = agent.householdtime;
	agents->churchtime[index] = agent.churchtime;
	agents->transporttime[index] = agent.transporttime;
	agents->clinictime[index] = agent.clinictime;
	agents->workplacetime[index] = agent.workplacetime;
	agents->outsidetime[index] = agent.outsidetime;
	agents->age[index] = agent.age;
	agents->gender[index] = agent.gender;
	agents->householdsize[index] = agent.householdsize;
	agents->churchfreq[index] = agent.churchfreq;
	agents->churchdur[index] = agent.churchdur;
	agents->transportdur[index] = agent.transportdur;
	agents->transportday1[index] = agent.transportday1;
	agents->transportday2[index] = agent.transportday2;
	agents->household[index] = agent.household;
	agents->church[index] = agent.church;
	agents->transport[index] = agent.transport;
	agents->workplace[index] = agent.workplace;
	agents->busy[index] = agent.busy;
	agents->startstep[index] = agent.startstep;
	agents->location[index] = agent.location;
	agents->locationid[index] = agent.locationid;
	agents->hiv[index] = agent.hiv;
	agents->art[index] = agent.art;
	agents->activetb[index] = agent.activetb;
	agents->artday[index] = agent.artday;
	agents->p[index] = agent.p;
	agents->q[index] = agent.q;
	agents->infections[index] = agent.infections;
	agents->lastinfected[index] = agent.lastinfected;
	agents->lastinfectedid[index] = agent.lastinfectedid;
	agents->time_step[index] = agent.time_step;
	agents->lambda[index] = agent.lambda;
	agents->timevisiting[index] = agent.timevisiting;
	agents->bargoing[index] = agent.bargoing;
	agents->barday[index] = agent.barday;
	}
}

/**
 *
 */
__global__ void GPUFLAME_tbinit(xmachine_memory_TBAssignment_list* agents, xmachine_message_tb_assignment_list* tb_assignment_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_TBAssignment_count)
        return;
    

	//SoA to AoS - xmachine_memory_tbinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_TBAssignment agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];

	//FLAME function call
	int dead = !tbinit(&agent, tb_assignment_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_tbinit Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
}

/**
 *
 */
__global__ void GPUFLAME_hhupdate(xmachine_memory_Household_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_hhupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Household agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Household_count){
    
	agent.id = agents->id[index];
	agent.lambda = agents->lambda[index];
	agent.active = agents->active[index];
	} else {
	
	agent.id = 0;
	agent.lambda = 0;
	agent.active = 0;
	}

	//FLAME function call
	int dead = !hhupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Household_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_hhupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->lambda[index] = agent.lambda;
	agents->active[index] = agent.active;
	}
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
	agent.household_size = agents->household_size[index];
	agent.churchgoing = agents->churchgoing[index];
	agent.churchfreq = agents->churchfreq[index];
	} else {
	
	agent.household_id = 0;
	agent.person_id = 0;
	agent.household_size = 0;
	agent.churchgoing = 0;
	agent.churchfreq = 0;
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
	agents->household_size[index] = agent.household_size;
	agents->churchgoing[index] = agent.churchgoing;
	agents->churchfreq[index] = agent.churchfreq;
	}
}

/**
 *
 */
__global__ void GPUFLAME_chuupdate(xmachine_memory_Church_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_chuupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Church agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Church_count){
    
	agent.id = agents->id[index];
	agent.size = agents->size[index];
	agent.lambda = agents->lambda[index];
	agent.active = agents->active[index];
	} else {
	
	agent.id = 0;
	agent.size = 0;
	agent.lambda = 0;
	agent.active = 0;
	}

	//FLAME function call
	int dead = !chuupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Church_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_chuupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->size[index] = agent.size;
	agents->lambda[index] = agent.lambda;
	agents->active[index] = agent.active;
	}
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
	agent.churchdur = agents->churchdur[index];

	//FLAME function call
	int dead = !chuinit(&agent, church_membership_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_chuinit Coalesced memory write (ignore arrays)
	agents->church_id[index] = agent.church_id;
	agents->household_id[index] = agent.household_id;
	agents->churchdur[index] = agent.churchdur;
}

/**
 *
 */
__global__ void GPUFLAME_trupdate(xmachine_memory_Transport_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_trupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Transport agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Transport_count){
    
	agent.id = agents->id[index];
	agent.lambda = agents->lambda[index];
	agent.active = agents->active[index];
	} else {
	
	agent.id = 0;
	agent.lambda = 0;
	agent.active = 0;
	}

	//FLAME function call
	int dead = !trupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Transport_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_trupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->lambda[index] = agent.lambda;
	agents->active[index] = agent.active;
	}
}

/**
 *
 */
__global__ void GPUFLAME_trinit(xmachine_memory_TransportMembership_list* agents, xmachine_message_transport_membership_list* transport_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_TransportMembership_count)
        return;
    

	//SoA to AoS - xmachine_memory_trinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_TransportMembership agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.person_id = agents->person_id[index];
	agent.transport_id = agents->transport_id[index];
	agent.duration = agents->duration[index];

	//FLAME function call
	int dead = !trinit(&agent, transport_membership_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_trinit Coalesced memory write (ignore arrays)
	agents->person_id[index] = agent.person_id;
	agents->transport_id[index] = agent.transport_id;
	agents->duration[index] = agent.duration;
}

/**
 *
 */
__global__ void GPUFLAME_clupdate(xmachine_memory_Clinic_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_clupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Clinic agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Clinic_count){
    
	agent.id = agents->id[index];
	agent.lambda = agents->lambda[index];
	} else {
	
	agent.id = 0;
	agent.lambda = 0;
	}

	//FLAME function call
	int dead = !clupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Clinic_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_clupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->lambda[index] = agent.lambda;
	}
}

/**
 *
 */
__global__ void GPUFLAME_wpupdate(xmachine_memory_Workplace_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_wpupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Workplace agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Workplace_count){
    
	agent.id = agents->id[index];
	agent.lambda = agents->lambda[index];
	} else {
	
	agent.id = 0;
	agent.lambda = 0;
	}

	//FLAME function call
	int dead = !wpupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Workplace_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_wpupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->lambda[index] = agent.lambda;
	}
}

/**
 *
 */
__global__ void GPUFLAME_wpinit(xmachine_memory_WorkplaceMembership_list* agents, xmachine_message_workplace_membership_list* workplace_membership_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_WorkplaceMembership_count)
        return;
    

	//SoA to AoS - xmachine_memory_wpinit Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_WorkplaceMembership agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.person_id = agents->person_id[index];
	agent.workplace_id = agents->workplace_id[index];

	//FLAME function call
	int dead = !wpinit(&agent, workplace_membership_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_wpinit Coalesced memory write (ignore arrays)
	agents->person_id[index] = agent.person_id;
	agents->workplace_id[index] = agent.workplace_id;
}

/**
 *
 */
__global__ void GPUFLAME_bupdate(xmachine_memory_Bar_list* agents, xmachine_message_location_list* location_messages, xmachine_message_infection_list* infection_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_bupdate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Bar agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_Bar_count){
    
	agent.id = agents->id[index];
	agent.lambda = agents->lambda[index];
	} else {
	
	agent.id = 0;
	agent.lambda = 0;
	}

	//FLAME function call
	int dead = !bupdate(&agent, location_messages, infection_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_Bar_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_bupdate Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->lambda[index] = agent.lambda;
	}
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
