
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
#define buffer_size_MAX 262144

//Maximum population size of xmachine_memory_Person
#define xmachine_memory_Person_MAX 262144

//Maximum population size of xmachine_memory_TBAssignment
#define xmachine_memory_TBAssignment_MAX 262144

//Maximum population size of xmachine_memory_Household
#define xmachine_memory_Household_MAX 262144

//Maximum population size of xmachine_memory_HouseholdMembership
#define xmachine_memory_HouseholdMembership_MAX 262144

//Maximum population size of xmachine_memory_Church
#define xmachine_memory_Church_MAX 262144

//Maximum population size of xmachine_memory_ChurchMembership
#define xmachine_memory_ChurchMembership_MAX 262144

//Maximum population size of xmachine_memory_Transport
#define xmachine_memory_Transport_MAX 262144

//Maximum population size of xmachine_memory_TransportMembership
#define xmachine_memory_TransportMembership_MAX 262144

//Maximum population size of xmachine_memory_Clinic
#define xmachine_memory_Clinic_MAX 262144

//Maximum population size of xmachine_memory_Workplace
#define xmachine_memory_Workplace_MAX 262144

//Maximum population size of xmachine_memory_WorkplaceMembership
#define xmachine_memory_WorkplaceMembership_MAX 262144

//Maximum population size of xmachine_memory_Bar
#define xmachine_memory_Bar_MAX 262144

//Maximum population size of xmachine_memory_School
#define xmachine_memory_School_MAX 262144

//Maximum population size of xmachine_memory_SchoolMembership
#define xmachine_memory_SchoolMembership_MAX 262144


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_tb_assignment
#define xmachine_message_tb_assignment_MAX 262144

//Maximum population size of xmachine_mmessage_household_membership
#define xmachine_message_household_membership_MAX 262144

//Maximum population size of xmachine_mmessage_church_membership
#define xmachine_message_church_membership_MAX 262144

//Maximum population size of xmachine_mmessage_transport_membership
#define xmachine_message_transport_membership_MAX 262144

//Maximum population size of xmachine_mmessage_workplace_membership
#define xmachine_message_workplace_membership_MAX 262144

//Maximum population size of xmachine_mmessage_school_membership
#define xmachine_message_school_membership_MAX 262144

//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX 262144

//Maximum population size of xmachine_mmessage_household_infection
#define xmachine_message_household_infection_MAX 262144

//Maximum population size of xmachine_mmessage_church_infection
#define xmachine_message_church_infection_MAX 262144

//Maximum population size of xmachine_mmessage_transport_infection
#define xmachine_message_transport_infection_MAX 262144

//Maximum population size of xmachine_mmessage_clinic_infection
#define xmachine_message_clinic_infection_MAX 262144

//Maximum population size of xmachine_mmessage_workplace_infection
#define xmachine_message_workplace_infection_MAX 262144

//Maximum population size of xmachine_mmessage_bar_infection
#define xmachine_message_bar_infection_MAX 262144

//Maximum population size of xmachine_mmessage_school_infection
#define xmachine_message_school_infection_MAX 262144


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_tb_assignment_partitioningNone
#define xmachine_message_household_membership_partitioningNone
#define xmachine_message_church_membership_partitioningNone
#define xmachine_message_transport_membership_partitioningNone
#define xmachine_message_workplace_membership_partitioningNone
#define xmachine_message_school_membership_partitioningNone
#define xmachine_message_location_partitioningNone
#define xmachine_message_household_infection_partitioningNone
#define xmachine_message_church_infection_partitioningNone
#define xmachine_message_transport_infection_partitioningNone
#define xmachine_message_clinic_infection_partitioningNone
#define xmachine_message_workplace_infection_partitioningNone
#define xmachine_message_bar_infection_partitioningNone
#define xmachine_message_school_infection_partitioningNone

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
    unsigned int step;    /**< X-machine memory variable step of type unsigned int.*/
    unsigned int householdtime;    /**< X-machine memory variable householdtime of type unsigned int.*/
    unsigned int churchtime;    /**< X-machine memory variable churchtime of type unsigned int.*/
    unsigned int transporttime;    /**< X-machine memory variable transporttime of type unsigned int.*/
    unsigned int clinictime;    /**< X-machine memory variable clinictime of type unsigned int.*/
    unsigned int workplacetime;    /**< X-machine memory variable workplacetime of type unsigned int.*/
    unsigned int bartime;    /**< X-machine memory variable bartime of type unsigned int.*/
    unsigned int outsidetime;    /**< X-machine memory variable outsidetime of type unsigned int.*/
    unsigned int age;    /**< X-machine memory variable age of type unsigned int.*/
    unsigned int gender;    /**< X-machine memory variable gender of type unsigned int.*/
    unsigned int householdsize;    /**< X-machine memory variable householdsize of type unsigned int.*/
    unsigned int churchfreq;    /**< X-machine memory variable churchfreq of type unsigned int.*/
    float churchdur;    /**< X-machine memory variable churchdur of type float.*/
    unsigned int transportdur;    /**< X-machine memory variable transportdur of type unsigned int.*/
    int transportday1;    /**< X-machine memory variable transportday1 of type int.*/
    int transportday2;    /**< X-machine memory variable transportday2 of type int.*/
    unsigned int household;    /**< X-machine memory variable household of type unsigned int.*/
    int church;    /**< X-machine memory variable church of type int.*/
    int transport;    /**< X-machine memory variable transport of type int.*/
    int workplace;    /**< X-machine memory variable workplace of type int.*/
    int school;    /**< X-machine memory variable school of type int.*/
    unsigned int busy;    /**< X-machine memory variable busy of type unsigned int.*/
    unsigned int startstep;    /**< X-machine memory variable startstep of type unsigned int.*/
    unsigned int location;    /**< X-machine memory variable location of type unsigned int.*/
    unsigned int locationid;    /**< X-machine memory variable locationid of type unsigned int.*/
    unsigned int hiv;    /**< X-machine memory variable hiv of type unsigned int.*/
    unsigned int art;    /**< X-machine memory variable art of type unsigned int.*/
    unsigned int activetb;    /**< X-machine memory variable activetb of type unsigned int.*/
    unsigned int artday;    /**< X-machine memory variable artday of type unsigned int.*/
    float p;    /**< X-machine memory variable p of type float.*/
    float q;    /**< X-machine memory variable q of type float.*/
    unsigned int infections;    /**< X-machine memory variable infections of type unsigned int.*/
    int lastinfected;    /**< X-machine memory variable lastinfected of type int.*/
    int lastinfectedid;    /**< X-machine memory variable lastinfectedid of type int.*/
    int lastinfectedtime;    /**< X-machine memory variable lastinfectedtime of type int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
    unsigned int timevisiting;    /**< X-machine memory variable timevisiting of type unsigned int.*/
    unsigned int bargoing;    /**< X-machine memory variable bargoing of type unsigned int.*/
    unsigned int barday;    /**< X-machine memory variable barday of type unsigned int.*/
    unsigned int schooltime;    /**< X-machine memory variable schooltime of type unsigned int.*/
};

/** struct xmachine_memory_TBAssignment
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_TBAssignment
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
};

/** struct xmachine_memory_Household
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Household
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
    unsigned int active;    /**< X-machine memory variable active of type unsigned int.*/
};

/** struct xmachine_memory_HouseholdMembership
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_HouseholdMembership
{
    unsigned int household_id;    /**< X-machine memory variable household_id of type unsigned int.*/
    unsigned int person_id;    /**< X-machine memory variable person_id of type unsigned int.*/
    unsigned int household_size;    /**< X-machine memory variable household_size of type unsigned int.*/
    unsigned int churchgoing;    /**< X-machine memory variable churchgoing of type unsigned int.*/
    unsigned int churchfreq;    /**< X-machine memory variable churchfreq of type unsigned int.*/
};

/** struct xmachine_memory_Church
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Church
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
    unsigned int active;    /**< X-machine memory variable active of type unsigned int.*/
};

/** struct xmachine_memory_ChurchMembership
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_ChurchMembership
{
    unsigned int church_id;    /**< X-machine memory variable church_id of type unsigned int.*/
    unsigned int household_id;    /**< X-machine memory variable household_id of type unsigned int.*/
    float churchdur;    /**< X-machine memory variable churchdur of type float.*/
};

/** struct xmachine_memory_Transport
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Transport
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
    unsigned int active;    /**< X-machine memory variable active of type unsigned int.*/
};

/** struct xmachine_memory_TransportMembership
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_TransportMembership
{
    int person_id;    /**< X-machine memory variable person_id of type int.*/
    unsigned int transport_id;    /**< X-machine memory variable transport_id of type unsigned int.*/
    unsigned int duration;    /**< X-machine memory variable duration of type unsigned int.*/
};

/** struct xmachine_memory_Clinic
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Clinic
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
};

/** struct xmachine_memory_Workplace
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Workplace
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
};

/** struct xmachine_memory_WorkplaceMembership
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_WorkplaceMembership
{
    unsigned int person_id;    /**< X-machine memory variable person_id of type unsigned int.*/
    unsigned int workplace_id;    /**< X-machine memory variable workplace_id of type unsigned int.*/
};

/** struct xmachine_memory_Bar
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Bar
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
};

/** struct xmachine_memory_School
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_School
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float lambda;    /**< X-machine memory variable lambda of type float.*/
};

/** struct xmachine_memory_SchoolMembership
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_SchoolMembership
{
    unsigned int person_id;    /**< X-machine memory variable person_id of type unsigned int.*/
    unsigned int school_id;    /**< X-machine memory variable school_id of type unsigned int.*/
};



/* Message structures */

/** struct xmachine_message_tb_assignment
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_tb_assignment
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_household_membership
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_household_membership
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int household_id;        /**< Message variable household_id of type unsigned int.*/  
    unsigned int person_id;        /**< Message variable person_id of type unsigned int.*/  
    unsigned int household_size;        /**< Message variable household_size of type unsigned int.*/  
    unsigned int church_id;        /**< Message variable church_id of type unsigned int.*/  
    unsigned int churchfreq;        /**< Message variable churchfreq of type unsigned int.*/  
    float churchdur;        /**< Message variable churchdur of type float.*/
};

/** struct xmachine_message_church_membership
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_church_membership
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int church_id;        /**< Message variable church_id of type unsigned int.*/  
    unsigned int household_id;        /**< Message variable household_id of type unsigned int.*/  
    float churchdur;        /**< Message variable churchdur of type float.*/
};

/** struct xmachine_message_transport_membership
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_transport_membership
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int person_id;        /**< Message variable person_id of type unsigned int.*/  
    unsigned int transport_id;        /**< Message variable transport_id of type unsigned int.*/  
    unsigned int duration;        /**< Message variable duration of type unsigned int.*/
};

/** struct xmachine_message_workplace_membership
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_workplace_membership
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int person_id;        /**< Message variable person_id of type unsigned int.*/  
    unsigned int workplace_id;        /**< Message variable workplace_id of type unsigned int.*/
};

/** struct xmachine_message_school_membership
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_school_membership
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int person_id;        /**< Message variable person_id of type unsigned int.*/  
    unsigned int school_id;        /**< Message variable school_id of type unsigned int.*/
};

/** struct xmachine_message_location
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int person_id;        /**< Message variable person_id of type unsigned int.*/  
    unsigned int location;        /**< Message variable location of type unsigned int.*/  
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float p;        /**< Message variable p of type float.*/  
    float q;        /**< Message variable q of type float.*/
};

/** struct xmachine_message_household_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_household_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_church_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_church_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_transport_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_transport_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_clinic_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_clinic_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_workplace_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_workplace_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_bar_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_bar_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};

/** struct xmachine_message_school_infection
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_school_infection
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int locationid;        /**< Message variable locationid of type unsigned int.*/  
    float lambda;        /**< Message variable lambda of type float.*/
};



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
    unsigned int step [xmachine_memory_Person_MAX];    /**< X-machine memory variable list step of type unsigned int.*/
    unsigned int householdtime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list householdtime of type unsigned int.*/
    unsigned int churchtime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list churchtime of type unsigned int.*/
    unsigned int transporttime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list transporttime of type unsigned int.*/
    unsigned int clinictime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list clinictime of type unsigned int.*/
    unsigned int workplacetime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list workplacetime of type unsigned int.*/
    unsigned int bartime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list bartime of type unsigned int.*/
    unsigned int outsidetime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list outsidetime of type unsigned int.*/
    unsigned int age [xmachine_memory_Person_MAX];    /**< X-machine memory variable list age of type unsigned int.*/
    unsigned int gender [xmachine_memory_Person_MAX];    /**< X-machine memory variable list gender of type unsigned int.*/
    unsigned int householdsize [xmachine_memory_Person_MAX];    /**< X-machine memory variable list householdsize of type unsigned int.*/
    unsigned int churchfreq [xmachine_memory_Person_MAX];    /**< X-machine memory variable list churchfreq of type unsigned int.*/
    float churchdur [xmachine_memory_Person_MAX];    /**< X-machine memory variable list churchdur of type float.*/
    unsigned int transportdur [xmachine_memory_Person_MAX];    /**< X-machine memory variable list transportdur of type unsigned int.*/
    int transportday1 [xmachine_memory_Person_MAX];    /**< X-machine memory variable list transportday1 of type int.*/
    int transportday2 [xmachine_memory_Person_MAX];    /**< X-machine memory variable list transportday2 of type int.*/
    unsigned int household [xmachine_memory_Person_MAX];    /**< X-machine memory variable list household of type unsigned int.*/
    int church [xmachine_memory_Person_MAX];    /**< X-machine memory variable list church of type int.*/
    int transport [xmachine_memory_Person_MAX];    /**< X-machine memory variable list transport of type int.*/
    int workplace [xmachine_memory_Person_MAX];    /**< X-machine memory variable list workplace of type int.*/
    int school [xmachine_memory_Person_MAX];    /**< X-machine memory variable list school of type int.*/
    unsigned int busy [xmachine_memory_Person_MAX];    /**< X-machine memory variable list busy of type unsigned int.*/
    unsigned int startstep [xmachine_memory_Person_MAX];    /**< X-machine memory variable list startstep of type unsigned int.*/
    unsigned int location [xmachine_memory_Person_MAX];    /**< X-machine memory variable list location of type unsigned int.*/
    unsigned int locationid [xmachine_memory_Person_MAX];    /**< X-machine memory variable list locationid of type unsigned int.*/
    unsigned int hiv [xmachine_memory_Person_MAX];    /**< X-machine memory variable list hiv of type unsigned int.*/
    unsigned int art [xmachine_memory_Person_MAX];    /**< X-machine memory variable list art of type unsigned int.*/
    unsigned int activetb [xmachine_memory_Person_MAX];    /**< X-machine memory variable list activetb of type unsigned int.*/
    unsigned int artday [xmachine_memory_Person_MAX];    /**< X-machine memory variable list artday of type unsigned int.*/
    float p [xmachine_memory_Person_MAX];    /**< X-machine memory variable list p of type float.*/
    float q [xmachine_memory_Person_MAX];    /**< X-machine memory variable list q of type float.*/
    unsigned int infections [xmachine_memory_Person_MAX];    /**< X-machine memory variable list infections of type unsigned int.*/
    int lastinfected [xmachine_memory_Person_MAX];    /**< X-machine memory variable list lastinfected of type int.*/
    int lastinfectedid [xmachine_memory_Person_MAX];    /**< X-machine memory variable list lastinfectedid of type int.*/
    int lastinfectedtime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list lastinfectedtime of type int.*/
    float lambda [xmachine_memory_Person_MAX];    /**< X-machine memory variable list lambda of type float.*/
    unsigned int timevisiting [xmachine_memory_Person_MAX];    /**< X-machine memory variable list timevisiting of type unsigned int.*/
    unsigned int bargoing [xmachine_memory_Person_MAX];    /**< X-machine memory variable list bargoing of type unsigned int.*/
    unsigned int barday [xmachine_memory_Person_MAX];    /**< X-machine memory variable list barday of type unsigned int.*/
    unsigned int schooltime [xmachine_memory_Person_MAX];    /**< X-machine memory variable list schooltime of type unsigned int.*/
};

/** struct xmachine_memory_TBAssignment_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_TBAssignment_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_TBAssignment_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_TBAssignment_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_TBAssignment_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
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
    float lambda [xmachine_memory_Household_MAX];    /**< X-machine memory variable list lambda of type float.*/
    unsigned int active [xmachine_memory_Household_MAX];    /**< X-machine memory variable list active of type unsigned int.*/
};

/** struct xmachine_memory_HouseholdMembership_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_HouseholdMembership_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_HouseholdMembership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_HouseholdMembership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int household_id [xmachine_memory_HouseholdMembership_MAX];    /**< X-machine memory variable list household_id of type unsigned int.*/
    unsigned int person_id [xmachine_memory_HouseholdMembership_MAX];    /**< X-machine memory variable list person_id of type unsigned int.*/
    unsigned int household_size [xmachine_memory_HouseholdMembership_MAX];    /**< X-machine memory variable list household_size of type unsigned int.*/
    unsigned int churchgoing [xmachine_memory_HouseholdMembership_MAX];    /**< X-machine memory variable list churchgoing of type unsigned int.*/
    unsigned int churchfreq [xmachine_memory_HouseholdMembership_MAX];    /**< X-machine memory variable list churchfreq of type unsigned int.*/
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
    float lambda [xmachine_memory_Church_MAX];    /**< X-machine memory variable list lambda of type float.*/
    unsigned int active [xmachine_memory_Church_MAX];    /**< X-machine memory variable list active of type unsigned int.*/
};

/** struct xmachine_memory_ChurchMembership_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_ChurchMembership_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_ChurchMembership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_ChurchMembership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int church_id [xmachine_memory_ChurchMembership_MAX];    /**< X-machine memory variable list church_id of type unsigned int.*/
    unsigned int household_id [xmachine_memory_ChurchMembership_MAX];    /**< X-machine memory variable list household_id of type unsigned int.*/
    float churchdur [xmachine_memory_ChurchMembership_MAX];    /**< X-machine memory variable list churchdur of type float.*/
};

/** struct xmachine_memory_Transport_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Transport_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Transport_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Transport_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Transport_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float lambda [xmachine_memory_Transport_MAX];    /**< X-machine memory variable list lambda of type float.*/
    unsigned int active [xmachine_memory_Transport_MAX];    /**< X-machine memory variable list active of type unsigned int.*/
};

/** struct xmachine_memory_TransportMembership_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_TransportMembership_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_TransportMembership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_TransportMembership_MAX];  /**< Used during parallel prefix sum */
    
    int person_id [xmachine_memory_TransportMembership_MAX];    /**< X-machine memory variable list person_id of type int.*/
    unsigned int transport_id [xmachine_memory_TransportMembership_MAX];    /**< X-machine memory variable list transport_id of type unsigned int.*/
    unsigned int duration [xmachine_memory_TransportMembership_MAX];    /**< X-machine memory variable list duration of type unsigned int.*/
};

/** struct xmachine_memory_Clinic_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Clinic_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Clinic_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Clinic_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Clinic_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float lambda [xmachine_memory_Clinic_MAX];    /**< X-machine memory variable list lambda of type float.*/
};

/** struct xmachine_memory_Workplace_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Workplace_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Workplace_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Workplace_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Workplace_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float lambda [xmachine_memory_Workplace_MAX];    /**< X-machine memory variable list lambda of type float.*/
};

/** struct xmachine_memory_WorkplaceMembership_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_WorkplaceMembership_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_WorkplaceMembership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_WorkplaceMembership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_memory_WorkplaceMembership_MAX];    /**< X-machine memory variable list person_id of type unsigned int.*/
    unsigned int workplace_id [xmachine_memory_WorkplaceMembership_MAX];    /**< X-machine memory variable list workplace_id of type unsigned int.*/
};

/** struct xmachine_memory_Bar_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Bar_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Bar_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Bar_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Bar_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float lambda [xmachine_memory_Bar_MAX];    /**< X-machine memory variable list lambda of type float.*/
};

/** struct xmachine_memory_School_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_School_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_School_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_School_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_School_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float lambda [xmachine_memory_School_MAX];    /**< X-machine memory variable list lambda of type float.*/
};

/** struct xmachine_memory_SchoolMembership_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_SchoolMembership_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_SchoolMembership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_SchoolMembership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_memory_SchoolMembership_MAX];    /**< X-machine memory variable list person_id of type unsigned int.*/
    unsigned int school_id [xmachine_memory_SchoolMembership_MAX];    /**< X-machine memory variable list school_id of type unsigned int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_tb_assignment_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_tb_assignment_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_tb_assignment_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_tb_assignment_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_tb_assignment_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_household_membership_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_household_membership_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_household_membership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_household_membership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int household_id [xmachine_message_household_membership_MAX];    /**< Message memory variable list household_id of type unsigned int.*/
    unsigned int person_id [xmachine_message_household_membership_MAX];    /**< Message memory variable list person_id of type unsigned int.*/
    unsigned int household_size [xmachine_message_household_membership_MAX];    /**< Message memory variable list household_size of type unsigned int.*/
    unsigned int church_id [xmachine_message_household_membership_MAX];    /**< Message memory variable list church_id of type unsigned int.*/
    unsigned int churchfreq [xmachine_message_household_membership_MAX];    /**< Message memory variable list churchfreq of type unsigned int.*/
    float churchdur [xmachine_message_household_membership_MAX];    /**< Message memory variable list churchdur of type float.*/
    
};

/** struct xmachine_message_church_membership_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_church_membership_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_church_membership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_church_membership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int church_id [xmachine_message_church_membership_MAX];    /**< Message memory variable list church_id of type unsigned int.*/
    unsigned int household_id [xmachine_message_church_membership_MAX];    /**< Message memory variable list household_id of type unsigned int.*/
    float churchdur [xmachine_message_church_membership_MAX];    /**< Message memory variable list churchdur of type float.*/
    
};

/** struct xmachine_message_transport_membership_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_transport_membership_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_transport_membership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_transport_membership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_message_transport_membership_MAX];    /**< Message memory variable list person_id of type unsigned int.*/
    unsigned int transport_id [xmachine_message_transport_membership_MAX];    /**< Message memory variable list transport_id of type unsigned int.*/
    unsigned int duration [xmachine_message_transport_membership_MAX];    /**< Message memory variable list duration of type unsigned int.*/
    
};

/** struct xmachine_message_workplace_membership_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_workplace_membership_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_workplace_membership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_workplace_membership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_message_workplace_membership_MAX];    /**< Message memory variable list person_id of type unsigned int.*/
    unsigned int workplace_id [xmachine_message_workplace_membership_MAX];    /**< Message memory variable list workplace_id of type unsigned int.*/
    
};

/** struct xmachine_message_school_membership_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_school_membership_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_school_membership_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_school_membership_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_message_school_membership_MAX];    /**< Message memory variable list person_id of type unsigned int.*/
    unsigned int school_id [xmachine_message_school_membership_MAX];    /**< Message memory variable list school_id of type unsigned int.*/
    
};

/** struct xmachine_message_location_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int person_id [xmachine_message_location_MAX];    /**< Message memory variable list person_id of type unsigned int.*/
    unsigned int location [xmachine_message_location_MAX];    /**< Message memory variable list location of type unsigned int.*/
    unsigned int locationid [xmachine_message_location_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float p [xmachine_message_location_MAX];    /**< Message memory variable list p of type float.*/
    float q [xmachine_message_location_MAX];    /**< Message memory variable list q of type float.*/
    
};

/** struct xmachine_message_household_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_household_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_household_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_household_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_household_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_household_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_church_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_church_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_church_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_church_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_church_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_church_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_transport_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_transport_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_transport_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_transport_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_transport_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_transport_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_clinic_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_clinic_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_clinic_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_clinic_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_clinic_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_clinic_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_workplace_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_workplace_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_workplace_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_workplace_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_workplace_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_workplace_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_bar_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_bar_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_bar_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_bar_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_bar_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_bar_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
};

/** struct xmachine_message_school_infection_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_school_infection_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_school_infection_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_school_infection_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int locationid [xmachine_message_school_infection_MAX];    /**< Message memory variable list locationid of type unsigned int.*/
    float lambda [xmachine_message_school_infection_MAX];    /**< Message memory variable list lambda of type float.*/
    
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
 * update FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int update(xmachine_memory_Person* agent, xmachine_message_location_list* location_messages, RNG_rand48* rand48);

/**
 * updatelambdahh FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param household_infection_messages  household_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_household_infection_message and get_next_household_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdahh(xmachine_memory_Person* agent, xmachine_message_household_infection_list* household_infection_messages);

/**
 * updatelambdachu FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param church_infection_messages  church_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_church_infection_message and get_next_church_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdachu(xmachine_memory_Person* agent, xmachine_message_church_infection_list* church_infection_messages);

/**
 * updatelambdatr FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param transport_infection_messages  transport_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_transport_infection_message and get_next_transport_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdatr(xmachine_memory_Person* agent, xmachine_message_transport_infection_list* transport_infection_messages);

/**
 * updatelambdacl FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param clinic_infection_messages  clinic_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_clinic_infection_message and get_next_clinic_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdacl(xmachine_memory_Person* agent, xmachine_message_clinic_infection_list* clinic_infection_messages);

/**
 * updatelambdawp FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param workplace_infection_messages  workplace_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_workplace_infection_message and get_next_workplace_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdawp(xmachine_memory_Person* agent, xmachine_message_workplace_infection_list* workplace_infection_messages);

/**
 * updatelambdab FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param bar_infection_messages  bar_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_bar_infection_message and get_next_bar_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdab(xmachine_memory_Person* agent, xmachine_message_bar_infection_list* bar_infection_messages);

/**
 * updatelambdasch FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param school_infection_messages  school_infection_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_school_infection_message and get_next_school_infection_message functions.
 */
__FLAME_GPU_FUNC__ int updatelambdasch(xmachine_memory_Person* agent, xmachine_message_school_infection_list* school_infection_messages);

/**
 * infect FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int infect(xmachine_memory_Person* agent, RNG_rand48* rand48);

/**
 * personhhinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param household_membership_messages  household_membership_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_household_membership_message and get_next_household_membership_message functions.
 */
__FLAME_GPU_FUNC__ int personhhinit(xmachine_memory_Person* agent, xmachine_message_household_membership_list* household_membership_messages);

/**
 * persontbinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param tb_assignment_messages  tb_assignment_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_tb_assignment_message and get_next_tb_assignment_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int persontbinit(xmachine_memory_Person* agent, xmachine_message_tb_assignment_list* tb_assignment_messages, RNG_rand48* rand48);

/**
 * persontrinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param transport_membership_messages  transport_membership_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_transport_membership_message and get_next_transport_membership_message functions.
 */
__FLAME_GPU_FUNC__ int persontrinit(xmachine_memory_Person* agent, xmachine_message_transport_membership_list* transport_membership_messages);

/**
 * personwpinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param workplace_membership_messages  workplace_membership_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_workplace_membership_message and get_next_workplace_membership_message functions.
 */
__FLAME_GPU_FUNC__ int personwpinit(xmachine_memory_Person* agent, xmachine_message_workplace_membership_list* workplace_membership_messages);

/**
 * personschinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Person. This represents a single agent instance and can be modified directly.
 * @param school_membership_messages  school_membership_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_school_membership_message and get_next_school_membership_message functions.
 */
__FLAME_GPU_FUNC__ int personschinit(xmachine_memory_Person* agent, xmachine_message_school_membership_list* school_membership_messages);

/**
 * tbinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_TBAssignment. This represents a single agent instance and can be modified directly.
 * @param tb_assignment_messages Pointer to output message list of type xmachine_message_tb_assignment_list. Must be passed as an argument to the add_tb_assignment_message function ??.
 */
__FLAME_GPU_FUNC__ int tbinit(xmachine_memory_TBAssignment* agent, xmachine_message_tb_assignment_list* tb_assignment_messages);

/**
 * hhupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Household. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param household_infection_messages Pointer to output message list of type xmachine_message_household_infection_list. Must be passed as an argument to the add_household_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int hhupdate(xmachine_memory_Household* agent, xmachine_message_location_list* location_messages, xmachine_message_household_infection_list* household_infection_messages);

/**
 * hhinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_HouseholdMembership. This represents a single agent instance and can be modified directly.
 * @param church_membership_messages  church_membership_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_church_membership_message and get_next_church_membership_message functions.* @param household_membership_messages Pointer to output message list of type xmachine_message_household_membership_list. Must be passed as an argument to the add_household_membership_message function ??.
 */
__FLAME_GPU_FUNC__ int hhinit(xmachine_memory_HouseholdMembership* agent, xmachine_message_church_membership_list* church_membership_messages, xmachine_message_household_membership_list* household_membership_messages);

/**
 * chuupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Church. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param church_infection_messages Pointer to output message list of type xmachine_message_church_infection_list. Must be passed as an argument to the add_church_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int chuupdate(xmachine_memory_Church* agent, xmachine_message_location_list* location_messages, xmachine_message_church_infection_list* church_infection_messages);

/**
 * chuinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_ChurchMembership. This represents a single agent instance and can be modified directly.
 * @param church_membership_messages Pointer to output message list of type xmachine_message_church_membership_list. Must be passed as an argument to the add_church_membership_message function ??.
 */
__FLAME_GPU_FUNC__ int chuinit(xmachine_memory_ChurchMembership* agent, xmachine_message_church_membership_list* church_membership_messages);

/**
 * trupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Transport. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param transport_infection_messages Pointer to output message list of type xmachine_message_transport_infection_list. Must be passed as an argument to the add_transport_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int trupdate(xmachine_memory_Transport* agent, xmachine_message_location_list* location_messages, xmachine_message_transport_infection_list* transport_infection_messages);

/**
 * trinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_TransportMembership. This represents a single agent instance and can be modified directly.
 * @param transport_membership_messages Pointer to output message list of type xmachine_message_transport_membership_list. Must be passed as an argument to the add_transport_membership_message function ??.
 */
__FLAME_GPU_FUNC__ int trinit(xmachine_memory_TransportMembership* agent, xmachine_message_transport_membership_list* transport_membership_messages);

/**
 * clupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Clinic. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param clinic_infection_messages Pointer to output message list of type xmachine_message_clinic_infection_list. Must be passed as an argument to the add_clinic_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int clupdate(xmachine_memory_Clinic* agent, xmachine_message_location_list* location_messages, xmachine_message_clinic_infection_list* clinic_infection_messages);

/**
 * wpupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Workplace. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param workplace_infection_messages Pointer to output message list of type xmachine_message_workplace_infection_list. Must be passed as an argument to the add_workplace_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int wpupdate(xmachine_memory_Workplace* agent, xmachine_message_location_list* location_messages, xmachine_message_workplace_infection_list* workplace_infection_messages);

/**
 * wpinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_WorkplaceMembership. This represents a single agent instance and can be modified directly.
 * @param workplace_membership_messages Pointer to output message list of type xmachine_message_workplace_membership_list. Must be passed as an argument to the add_workplace_membership_message function ??.
 */
__FLAME_GPU_FUNC__ int wpinit(xmachine_memory_WorkplaceMembership* agent, xmachine_message_workplace_membership_list* workplace_membership_messages);

/**
 * bupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Bar. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param bar_infection_messages Pointer to output message list of type xmachine_message_bar_infection_list. Must be passed as an argument to the add_bar_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int bupdate(xmachine_memory_Bar* agent, xmachine_message_location_list* location_messages, xmachine_message_bar_infection_list* bar_infection_messages);

/**
 * schupdate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_School. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param school_infection_messages Pointer to output message list of type xmachine_message_school_infection_list. Must be passed as an argument to the add_school_infection_message function ??.
 */
__FLAME_GPU_FUNC__ int schupdate(xmachine_memory_School* agent, xmachine_message_location_list* location_messages, xmachine_message_school_infection_list* school_infection_messages);

/**
 * schinit FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_SchoolMembership. This represents a single agent instance and can be modified directly.
 * @param school_membership_messages Pointer to output message list of type xmachine_message_school_membership_list. Must be passed as an argument to the add_school_membership_message function ??.
 */
__FLAME_GPU_FUNC__ int schinit(xmachine_memory_SchoolMembership* agent, xmachine_message_school_membership_list* school_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) tb_assignment message implemented in FLAMEGPU_Kernels */

/** add_tb_assignment_message
 * Function for all types of message partitioning
 * Adds a new tb_assignment agent to the xmachine_memory_tb_assignment_list list using a linear mapping
 * @param agents	xmachine_memory_tb_assignment_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_tb_assignment_message(xmachine_message_tb_assignment_list* tb_assignment_messages, unsigned int id);
 
/** get_first_tb_assignment_message
 * Get first message function for non partitioned (brute force) messages
 * @param tb_assignment_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_tb_assignment * get_first_tb_assignment_message(xmachine_message_tb_assignment_list* tb_assignment_messages);

/** get_next_tb_assignment_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param tb_assignment_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_tb_assignment * get_next_tb_assignment_message(xmachine_message_tb_assignment* current, xmachine_message_tb_assignment_list* tb_assignment_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) household_membership message implemented in FLAMEGPU_Kernels */

/** add_household_membership_message
 * Function for all types of message partitioning
 * Adds a new household_membership agent to the xmachine_memory_household_membership_list list using a linear mapping
 * @param agents	xmachine_memory_household_membership_list agent list
 * @param household_id	message variable of type unsigned int
 * @param person_id	message variable of type unsigned int
 * @param household_size	message variable of type unsigned int
 * @param church_id	message variable of type unsigned int
 * @param churchfreq	message variable of type unsigned int
 * @param churchdur	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_household_membership_message(xmachine_message_household_membership_list* household_membership_messages, unsigned int household_id, unsigned int person_id, unsigned int household_size, unsigned int church_id, unsigned int churchfreq, float churchdur);
 
/** get_first_household_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param household_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_household_membership * get_first_household_membership_message(xmachine_message_household_membership_list* household_membership_messages);

/** get_next_household_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param household_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_household_membership * get_next_household_membership_message(xmachine_message_household_membership* current, xmachine_message_household_membership_list* household_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) church_membership message implemented in FLAMEGPU_Kernels */

/** add_church_membership_message
 * Function for all types of message partitioning
 * Adds a new church_membership agent to the xmachine_memory_church_membership_list list using a linear mapping
 * @param agents	xmachine_memory_church_membership_list agent list
 * @param church_id	message variable of type unsigned int
 * @param household_id	message variable of type unsigned int
 * @param churchdur	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_church_membership_message(xmachine_message_church_membership_list* church_membership_messages, unsigned int church_id, unsigned int household_id, float churchdur);
 
/** get_first_church_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param church_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_church_membership * get_first_church_membership_message(xmachine_message_church_membership_list* church_membership_messages);

/** get_next_church_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param church_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_church_membership * get_next_church_membership_message(xmachine_message_church_membership* current, xmachine_message_church_membership_list* church_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) transport_membership message implemented in FLAMEGPU_Kernels */

/** add_transport_membership_message
 * Function for all types of message partitioning
 * Adds a new transport_membership agent to the xmachine_memory_transport_membership_list list using a linear mapping
 * @param agents	xmachine_memory_transport_membership_list agent list
 * @param person_id	message variable of type unsigned int
 * @param transport_id	message variable of type unsigned int
 * @param duration	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_transport_membership_message(xmachine_message_transport_membership_list* transport_membership_messages, unsigned int person_id, unsigned int transport_id, unsigned int duration);
 
/** get_first_transport_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param transport_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_transport_membership * get_first_transport_membership_message(xmachine_message_transport_membership_list* transport_membership_messages);

/** get_next_transport_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param transport_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_transport_membership * get_next_transport_membership_message(xmachine_message_transport_membership* current, xmachine_message_transport_membership_list* transport_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) workplace_membership message implemented in FLAMEGPU_Kernels */

/** add_workplace_membership_message
 * Function for all types of message partitioning
 * Adds a new workplace_membership agent to the xmachine_memory_workplace_membership_list list using a linear mapping
 * @param agents	xmachine_memory_workplace_membership_list agent list
 * @param person_id	message variable of type unsigned int
 * @param workplace_id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_workplace_membership_message(xmachine_message_workplace_membership_list* workplace_membership_messages, unsigned int person_id, unsigned int workplace_id);
 
/** get_first_workplace_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param workplace_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_workplace_membership * get_first_workplace_membership_message(xmachine_message_workplace_membership_list* workplace_membership_messages);

/** get_next_workplace_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param workplace_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_workplace_membership * get_next_workplace_membership_message(xmachine_message_workplace_membership* current, xmachine_message_workplace_membership_list* workplace_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) school_membership message implemented in FLAMEGPU_Kernels */

/** add_school_membership_message
 * Function for all types of message partitioning
 * Adds a new school_membership agent to the xmachine_memory_school_membership_list list using a linear mapping
 * @param agents	xmachine_memory_school_membership_list agent list
 * @param person_id	message variable of type unsigned int
 * @param school_id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_school_membership_message(xmachine_message_school_membership_list* school_membership_messages, unsigned int person_id, unsigned int school_id);
 
/** get_first_school_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param school_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_school_membership * get_first_school_membership_message(xmachine_message_school_membership_list* school_membership_messages);

/** get_next_school_membership_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param school_membership_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_school_membership * get_next_school_membership_message(xmachine_message_school_membership* current, xmachine_message_school_membership_list* school_membership_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param person_id	message variable of type unsigned int
 * @param location	message variable of type unsigned int
 * @param locationid	message variable of type unsigned int
 * @param p	message variable of type float
 * @param q	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, unsigned int person_id, unsigned int location, unsigned int locationid, float p, float q);
 
/** get_first_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages);

/** get_next_location_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param location_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) household_infection message implemented in FLAMEGPU_Kernels */

/** add_household_infection_message
 * Function for all types of message partitioning
 * Adds a new household_infection agent to the xmachine_memory_household_infection_list list using a linear mapping
 * @param agents	xmachine_memory_household_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_household_infection_message(xmachine_message_household_infection_list* household_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_household_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param household_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_household_infection * get_first_household_infection_message(xmachine_message_household_infection_list* household_infection_messages);

/** get_next_household_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param household_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_household_infection * get_next_household_infection_message(xmachine_message_household_infection* current, xmachine_message_household_infection_list* household_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) church_infection message implemented in FLAMEGPU_Kernels */

/** add_church_infection_message
 * Function for all types of message partitioning
 * Adds a new church_infection agent to the xmachine_memory_church_infection_list list using a linear mapping
 * @param agents	xmachine_memory_church_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_church_infection_message(xmachine_message_church_infection_list* church_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_church_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param church_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_church_infection * get_first_church_infection_message(xmachine_message_church_infection_list* church_infection_messages);

/** get_next_church_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param church_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_church_infection * get_next_church_infection_message(xmachine_message_church_infection* current, xmachine_message_church_infection_list* church_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) transport_infection message implemented in FLAMEGPU_Kernels */

/** add_transport_infection_message
 * Function for all types of message partitioning
 * Adds a new transport_infection agent to the xmachine_memory_transport_infection_list list using a linear mapping
 * @param agents	xmachine_memory_transport_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_transport_infection_message(xmachine_message_transport_infection_list* transport_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_transport_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param transport_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_transport_infection * get_first_transport_infection_message(xmachine_message_transport_infection_list* transport_infection_messages);

/** get_next_transport_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param transport_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_transport_infection * get_next_transport_infection_message(xmachine_message_transport_infection* current, xmachine_message_transport_infection_list* transport_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) clinic_infection message implemented in FLAMEGPU_Kernels */

/** add_clinic_infection_message
 * Function for all types of message partitioning
 * Adds a new clinic_infection agent to the xmachine_memory_clinic_infection_list list using a linear mapping
 * @param agents	xmachine_memory_clinic_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_clinic_infection_message(xmachine_message_clinic_infection_list* clinic_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_clinic_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param clinic_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_clinic_infection * get_first_clinic_infection_message(xmachine_message_clinic_infection_list* clinic_infection_messages);

/** get_next_clinic_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param clinic_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_clinic_infection * get_next_clinic_infection_message(xmachine_message_clinic_infection* current, xmachine_message_clinic_infection_list* clinic_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) workplace_infection message implemented in FLAMEGPU_Kernels */

/** add_workplace_infection_message
 * Function for all types of message partitioning
 * Adds a new workplace_infection agent to the xmachine_memory_workplace_infection_list list using a linear mapping
 * @param agents	xmachine_memory_workplace_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_workplace_infection_message(xmachine_message_workplace_infection_list* workplace_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_workplace_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param workplace_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_workplace_infection * get_first_workplace_infection_message(xmachine_message_workplace_infection_list* workplace_infection_messages);

/** get_next_workplace_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param workplace_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_workplace_infection * get_next_workplace_infection_message(xmachine_message_workplace_infection* current, xmachine_message_workplace_infection_list* workplace_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) bar_infection message implemented in FLAMEGPU_Kernels */

/** add_bar_infection_message
 * Function for all types of message partitioning
 * Adds a new bar_infection agent to the xmachine_memory_bar_infection_list list using a linear mapping
 * @param agents	xmachine_memory_bar_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_bar_infection_message(xmachine_message_bar_infection_list* bar_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_bar_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param bar_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_bar_infection * get_first_bar_infection_message(xmachine_message_bar_infection_list* bar_infection_messages);

/** get_next_bar_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param bar_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_bar_infection * get_next_bar_infection_message(xmachine_message_bar_infection* current, xmachine_message_bar_infection_list* bar_infection_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) school_infection message implemented in FLAMEGPU_Kernels */

/** add_school_infection_message
 * Function for all types of message partitioning
 * Adds a new school_infection agent to the xmachine_memory_school_infection_list list using a linear mapping
 * @param agents	xmachine_memory_school_infection_list agent list
 * @param locationid	message variable of type unsigned int
 * @param lambda	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_school_infection_message(xmachine_message_school_infection_list* school_infection_messages, unsigned int locationid, float lambda);
 
/** get_first_school_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param school_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_school_infection * get_first_school_infection_message(xmachine_message_school_infection_list* school_infection_messages);

/** get_next_school_infection_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param school_infection_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_school_infection * get_next_school_infection_message(xmachine_message_school_infection* current, xmachine_message_school_infection_list* school_infection_messages);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Person_agent
 * Adds a new continuous valued Person agent to the xmachine_memory_Person_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Person_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param step	agent agent variable of type unsigned int
 * @param householdtime	agent agent variable of type unsigned int
 * @param churchtime	agent agent variable of type unsigned int
 * @param transporttime	agent agent variable of type unsigned int
 * @param clinictime	agent agent variable of type unsigned int
 * @param workplacetime	agent agent variable of type unsigned int
 * @param bartime	agent agent variable of type unsigned int
 * @param outsidetime	agent agent variable of type unsigned int
 * @param age	agent agent variable of type unsigned int
 * @param gender	agent agent variable of type unsigned int
 * @param householdsize	agent agent variable of type unsigned int
 * @param churchfreq	agent agent variable of type unsigned int
 * @param churchdur	agent agent variable of type float
 * @param transportdur	agent agent variable of type unsigned int
 * @param transportday1	agent agent variable of type int
 * @param transportday2	agent agent variable of type int
 * @param household	agent agent variable of type unsigned int
 * @param church	agent agent variable of type int
 * @param transport	agent agent variable of type int
 * @param workplace	agent agent variable of type int
 * @param school	agent agent variable of type int
 * @param busy	agent agent variable of type unsigned int
 * @param startstep	agent agent variable of type unsigned int
 * @param location	agent agent variable of type unsigned int
 * @param locationid	agent agent variable of type unsigned int
 * @param hiv	agent agent variable of type unsigned int
 * @param art	agent agent variable of type unsigned int
 * @param activetb	agent agent variable of type unsigned int
 * @param artday	agent agent variable of type unsigned int
 * @param p	agent agent variable of type float
 * @param q	agent agent variable of type float
 * @param infections	agent agent variable of type unsigned int
 * @param lastinfected	agent agent variable of type int
 * @param lastinfectedid	agent agent variable of type int
 * @param lastinfectedtime	agent agent variable of type int
 * @param lambda	agent agent variable of type float
 * @param timevisiting	agent agent variable of type unsigned int
 * @param bargoing	agent agent variable of type unsigned int
 * @param barday	agent agent variable of type unsigned int
 * @param schooltime	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Person_agent(xmachine_memory_Person_list* agents, unsigned int id, unsigned int step, unsigned int householdtime, unsigned int churchtime, unsigned int transporttime, unsigned int clinictime, unsigned int workplacetime, unsigned int bartime, unsigned int outsidetime, unsigned int age, unsigned int gender, unsigned int householdsize, unsigned int churchfreq, float churchdur, unsigned int transportdur, int transportday1, int transportday2, unsigned int household, int church, int transport, int workplace, int school, unsigned int busy, unsigned int startstep, unsigned int location, unsigned int locationid, unsigned int hiv, unsigned int art, unsigned int activetb, unsigned int artday, float p, float q, unsigned int infections, int lastinfected, int lastinfectedid, int lastinfectedtime, float lambda, unsigned int timevisiting, unsigned int bargoing, unsigned int barday, unsigned int schooltime);

/** add_TBAssignment_agent
 * Adds a new continuous valued TBAssignment agent to the xmachine_memory_TBAssignment_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_TBAssignment_list agent list
 * @param id	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_TBAssignment_agent(xmachine_memory_TBAssignment_list* agents, unsigned int id);

/** add_Household_agent
 * Adds a new continuous valued Household agent to the xmachine_memory_Household_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Household_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 * @param active	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Household_agent(xmachine_memory_Household_list* agents, unsigned int id, float lambda, unsigned int active);

/** add_HouseholdMembership_agent
 * Adds a new continuous valued HouseholdMembership agent to the xmachine_memory_HouseholdMembership_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_HouseholdMembership_list agent list
 * @param household_id	agent agent variable of type unsigned int
 * @param person_id	agent agent variable of type unsigned int
 * @param household_size	agent agent variable of type unsigned int
 * @param churchgoing	agent agent variable of type unsigned int
 * @param churchfreq	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_HouseholdMembership_agent(xmachine_memory_HouseholdMembership_list* agents, unsigned int household_id, unsigned int person_id, unsigned int household_size, unsigned int churchgoing, unsigned int churchfreq);

/** add_Church_agent
 * Adds a new continuous valued Church agent to the xmachine_memory_Church_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Church_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 * @param active	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Church_agent(xmachine_memory_Church_list* agents, unsigned int id, unsigned int size, float lambda, unsigned int active);

/** add_ChurchMembership_agent
 * Adds a new continuous valued ChurchMembership agent to the xmachine_memory_ChurchMembership_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_ChurchMembership_list agent list
 * @param church_id	agent agent variable of type unsigned int
 * @param household_id	agent agent variable of type unsigned int
 * @param churchdur	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_ChurchMembership_agent(xmachine_memory_ChurchMembership_list* agents, unsigned int church_id, unsigned int household_id, float churchdur);

/** add_Transport_agent
 * Adds a new continuous valued Transport agent to the xmachine_memory_Transport_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Transport_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 * @param active	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_Transport_agent(xmachine_memory_Transport_list* agents, unsigned int id, float lambda, unsigned int active);

/** add_TransportMembership_agent
 * Adds a new continuous valued TransportMembership agent to the xmachine_memory_TransportMembership_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_TransportMembership_list agent list
 * @param person_id	agent agent variable of type int
 * @param transport_id	agent agent variable of type unsigned int
 * @param duration	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_TransportMembership_agent(xmachine_memory_TransportMembership_list* agents, int person_id, unsigned int transport_id, unsigned int duration);

/** add_Clinic_agent
 * Adds a new continuous valued Clinic agent to the xmachine_memory_Clinic_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Clinic_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Clinic_agent(xmachine_memory_Clinic_list* agents, unsigned int id, float lambda);

/** add_Workplace_agent
 * Adds a new continuous valued Workplace agent to the xmachine_memory_Workplace_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Workplace_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Workplace_agent(xmachine_memory_Workplace_list* agents, unsigned int id, float lambda);

/** add_WorkplaceMembership_agent
 * Adds a new continuous valued WorkplaceMembership agent to the xmachine_memory_WorkplaceMembership_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_WorkplaceMembership_list agent list
 * @param person_id	agent agent variable of type unsigned int
 * @param workplace_id	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_WorkplaceMembership_agent(xmachine_memory_WorkplaceMembership_list* agents, unsigned int person_id, unsigned int workplace_id);

/** add_Bar_agent
 * Adds a new continuous valued Bar agent to the xmachine_memory_Bar_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Bar_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Bar_agent(xmachine_memory_Bar_list* agents, unsigned int id, float lambda);

/** add_School_agent
 * Adds a new continuous valued School agent to the xmachine_memory_School_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_School_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param lambda	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_School_agent(xmachine_memory_School_list* agents, unsigned int id, float lambda);

/** add_SchoolMembership_agent
 * Adds a new continuous valued SchoolMembership agent to the xmachine_memory_SchoolMembership_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_SchoolMembership_list agent list
 * @param person_id	agent agent variable of type unsigned int
 * @param school_id	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_SchoolMembership_agent(xmachine_memory_SchoolMembership_list* agents, unsigned int person_id, unsigned int school_id);


  
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
 * @param h_TBAssignments Pointer to agent list on the host
 * @param d_TBAssignments Pointer to agent list on the GPU device
 * @param h_xmachine_memory_TBAssignment_count Pointer to agent counter
 * @param h_Households Pointer to agent list on the host
 * @param d_Households Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Household_count Pointer to agent counter
 * @param h_HouseholdMemberships Pointer to agent list on the host
 * @param d_HouseholdMemberships Pointer to agent list on the GPU device
 * @param h_xmachine_memory_HouseholdMembership_count Pointer to agent counter
 * @param h_Churchs Pointer to agent list on the host
 * @param d_Churchs Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Church_count Pointer to agent counter
 * @param h_ChurchMemberships Pointer to agent list on the host
 * @param d_ChurchMemberships Pointer to agent list on the GPU device
 * @param h_xmachine_memory_ChurchMembership_count Pointer to agent counter
 * @param h_Transports Pointer to agent list on the host
 * @param d_Transports Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Transport_count Pointer to agent counter
 * @param h_TransportMemberships Pointer to agent list on the host
 * @param d_TransportMemberships Pointer to agent list on the GPU device
 * @param h_xmachine_memory_TransportMembership_count Pointer to agent counter
 * @param h_Clinics Pointer to agent list on the host
 * @param d_Clinics Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Clinic_count Pointer to agent counter
 * @param h_Workplaces Pointer to agent list on the host
 * @param d_Workplaces Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Workplace_count Pointer to agent counter
 * @param h_WorkplaceMemberships Pointer to agent list on the host
 * @param d_WorkplaceMemberships Pointer to agent list on the GPU device
 * @param h_xmachine_memory_WorkplaceMembership_count Pointer to agent counter
 * @param h_Bars Pointer to agent list on the host
 * @param d_Bars Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Bar_count Pointer to agent counter
 * @param h_Schools Pointer to agent list on the host
 * @param d_Schools Pointer to agent list on the GPU device
 * @param h_xmachine_memory_School_count Pointer to agent counter
 * @param h_SchoolMemberships Pointer to agent list on the host
 * @param d_SchoolMemberships Pointer to agent list on the GPU device
 * @param h_xmachine_memory_SchoolMembership_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count,xmachine_memory_TBAssignment_list* h_TBAssignments_tbdefault, xmachine_memory_TBAssignment_list* d_TBAssignments_tbdefault, int h_xmachine_memory_TBAssignment_tbdefault_count,xmachine_memory_Household_list* h_Households_hhdefault, xmachine_memory_Household_list* d_Households_hhdefault, int h_xmachine_memory_Household_hhdefault_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships_hhmembershipdefault, xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_hhmembershipdefault, int h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count,xmachine_memory_Church_list* h_Churchs_chudefault, xmachine_memory_Church_list* d_Churchs_chudefault, int h_xmachine_memory_Church_chudefault_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships_chumembershipdefault, xmachine_memory_ChurchMembership_list* d_ChurchMemberships_chumembershipdefault, int h_xmachine_memory_ChurchMembership_chumembershipdefault_count,xmachine_memory_Transport_list* h_Transports_trdefault, xmachine_memory_Transport_list* d_Transports_trdefault, int h_xmachine_memory_Transport_trdefault_count,xmachine_memory_TransportMembership_list* h_TransportMemberships_trmembershipdefault, xmachine_memory_TransportMembership_list* d_TransportMemberships_trmembershipdefault, int h_xmachine_memory_TransportMembership_trmembershipdefault_count,xmachine_memory_Clinic_list* h_Clinics_cldefault, xmachine_memory_Clinic_list* d_Clinics_cldefault, int h_xmachine_memory_Clinic_cldefault_count,xmachine_memory_Workplace_list* h_Workplaces_wpdefault, xmachine_memory_Workplace_list* d_Workplaces_wpdefault, int h_xmachine_memory_Workplace_wpdefault_count,xmachine_memory_WorkplaceMembership_list* h_WorkplaceMemberships_wpmembershipdefault, xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_wpmembershipdefault, int h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count,xmachine_memory_Bar_list* h_Bars_bdefault, xmachine_memory_Bar_list* d_Bars_bdefault, int h_xmachine_memory_Bar_bdefault_count,xmachine_memory_School_list* h_Schools_schdefault, xmachine_memory_School_list* d_Schools_schdefault, int h_xmachine_memory_School_schdefault_count,xmachine_memory_SchoolMembership_list* h_SchoolMemberships_schmembershipdefault, xmachine_memory_SchoolMembership_list* d_SchoolMemberships_schmembershipdefault, int h_xmachine_memory_SchoolMembership_schmembershipdefault_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Persons Pointer to agent list on the host
 * @param h_xmachine_memory_Person_count Pointer to agent counter
 * @param h_TBAssignments Pointer to agent list on the host
 * @param h_xmachine_memory_TBAssignment_count Pointer to agent counter
 * @param h_Households Pointer to agent list on the host
 * @param h_xmachine_memory_Household_count Pointer to agent counter
 * @param h_HouseholdMemberships Pointer to agent list on the host
 * @param h_xmachine_memory_HouseholdMembership_count Pointer to agent counter
 * @param h_Churchs Pointer to agent list on the host
 * @param h_xmachine_memory_Church_count Pointer to agent counter
 * @param h_ChurchMemberships Pointer to agent list on the host
 * @param h_xmachine_memory_ChurchMembership_count Pointer to agent counter
 * @param h_Transports Pointer to agent list on the host
 * @param h_xmachine_memory_Transport_count Pointer to agent counter
 * @param h_TransportMemberships Pointer to agent list on the host
 * @param h_xmachine_memory_TransportMembership_count Pointer to agent counter
 * @param h_Clinics Pointer to agent list on the host
 * @param h_xmachine_memory_Clinic_count Pointer to agent counter
 * @param h_Workplaces Pointer to agent list on the host
 * @param h_xmachine_memory_Workplace_count Pointer to agent counter
 * @param h_WorkplaceMemberships Pointer to agent list on the host
 * @param h_xmachine_memory_WorkplaceMembership_count Pointer to agent counter
 * @param h_Bars Pointer to agent list on the host
 * @param h_xmachine_memory_Bar_count Pointer to agent counter
 * @param h_Schools Pointer to agent list on the host
 * @param h_xmachine_memory_School_count Pointer to agent counter
 * @param h_SchoolMemberships Pointer to agent list on the host
 * @param h_xmachine_memory_SchoolMembership_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count,xmachine_memory_TBAssignment_list* h_TBAssignments, int* h_xmachine_memory_TBAssignment_count,xmachine_memory_Household_list* h_Households, int* h_xmachine_memory_Household_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships, int* h_xmachine_memory_HouseholdMembership_count,xmachine_memory_Church_list* h_Churchs, int* h_xmachine_memory_Church_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships, int* h_xmachine_memory_ChurchMembership_count,xmachine_memory_Transport_list* h_Transports, int* h_xmachine_memory_Transport_count,xmachine_memory_TransportMembership_list* h_TransportMemberships, int* h_xmachine_memory_TransportMembership_count,xmachine_memory_Clinic_list* h_Clinics, int* h_xmachine_memory_Clinic_count,xmachine_memory_Workplace_list* h_Workplaces, int* h_xmachine_memory_Workplace_count,xmachine_memory_WorkplaceMembership_list* h_WorkplaceMemberships, int* h_xmachine_memory_WorkplaceMembership_count,xmachine_memory_Bar_list* h_Bars, int* h_xmachine_memory_Bar_count,xmachine_memory_School_list* h_Schools, int* h_xmachine_memory_School_count,xmachine_memory_SchoolMembership_list* h_SchoolMemberships, int* h_xmachine_memory_SchoolMembership_count);


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


    
/** get_agent_TBAssignment_MAX_count
 * Gets the max agent count for the TBAssignment agent type 
 * @return		the maximum TBAssignment agent count
 */
extern int get_agent_TBAssignment_MAX_count();



/** get_agent_TBAssignment_tbdefault_count
 * Gets the agent count for the TBAssignment agent type in state tbdefault
 * @return		the current TBAssignment agent count in state tbdefault
 */
extern int get_agent_TBAssignment_tbdefault_count();

/** reset_tbdefault_count
 * Resets the agent count of the TBAssignment in state tbdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_TBAssignment_tbdefault_count();

/** get_device_TBAssignment_tbdefault_agents
 * Gets a pointer to xmachine_memory_TBAssignment_list on the GPU device
 * @return		a xmachine_memory_TBAssignment_list on the GPU device
 */
extern xmachine_memory_TBAssignment_list* get_device_TBAssignment_tbdefault_agents();

/** get_host_TBAssignment_tbdefault_agents
 * Gets a pointer to xmachine_memory_TBAssignment_list on the CPU host
 * @return		a xmachine_memory_TBAssignment_list on the CPU host
 */
extern xmachine_memory_TBAssignment_list* get_host_TBAssignment_tbdefault_agents();


/** sort_TBAssignments_tbdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_TBAssignments_tbdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TBAssignment_list* agents));


    
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


    
/** get_agent_HouseholdMembership_MAX_count
 * Gets the max agent count for the HouseholdMembership agent type 
 * @return		the maximum HouseholdMembership agent count
 */
extern int get_agent_HouseholdMembership_MAX_count();



/** get_agent_HouseholdMembership_hhmembershipdefault_count
 * Gets the agent count for the HouseholdMembership agent type in state hhmembershipdefault
 * @return		the current HouseholdMembership agent count in state hhmembershipdefault
 */
extern int get_agent_HouseholdMembership_hhmembershipdefault_count();

/** reset_hhmembershipdefault_count
 * Resets the agent count of the HouseholdMembership in state hhmembershipdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_HouseholdMembership_hhmembershipdefault_count();

/** get_device_HouseholdMembership_hhmembershipdefault_agents
 * Gets a pointer to xmachine_memory_HouseholdMembership_list on the GPU device
 * @return		a xmachine_memory_HouseholdMembership_list on the GPU device
 */
extern xmachine_memory_HouseholdMembership_list* get_device_HouseholdMembership_hhmembershipdefault_agents();

/** get_host_HouseholdMembership_hhmembershipdefault_agents
 * Gets a pointer to xmachine_memory_HouseholdMembership_list on the CPU host
 * @return		a xmachine_memory_HouseholdMembership_list on the CPU host
 */
extern xmachine_memory_HouseholdMembership_list* get_host_HouseholdMembership_hhmembershipdefault_agents();


/** sort_HouseholdMemberships_hhmembershipdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_HouseholdMemberships_hhmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_HouseholdMembership_list* agents));


    
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


    
/** get_agent_ChurchMembership_MAX_count
 * Gets the max agent count for the ChurchMembership agent type 
 * @return		the maximum ChurchMembership agent count
 */
extern int get_agent_ChurchMembership_MAX_count();



/** get_agent_ChurchMembership_chumembershipdefault_count
 * Gets the agent count for the ChurchMembership agent type in state chumembershipdefault
 * @return		the current ChurchMembership agent count in state chumembershipdefault
 */
extern int get_agent_ChurchMembership_chumembershipdefault_count();

/** reset_chumembershipdefault_count
 * Resets the agent count of the ChurchMembership in state chumembershipdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_ChurchMembership_chumembershipdefault_count();

/** get_device_ChurchMembership_chumembershipdefault_agents
 * Gets a pointer to xmachine_memory_ChurchMembership_list on the GPU device
 * @return		a xmachine_memory_ChurchMembership_list on the GPU device
 */
extern xmachine_memory_ChurchMembership_list* get_device_ChurchMembership_chumembershipdefault_agents();

/** get_host_ChurchMembership_chumembershipdefault_agents
 * Gets a pointer to xmachine_memory_ChurchMembership_list on the CPU host
 * @return		a xmachine_memory_ChurchMembership_list on the CPU host
 */
extern xmachine_memory_ChurchMembership_list* get_host_ChurchMembership_chumembershipdefault_agents();


/** sort_ChurchMemberships_chumembershipdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_ChurchMemberships_chumembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_ChurchMembership_list* agents));


    
/** get_agent_Transport_MAX_count
 * Gets the max agent count for the Transport agent type 
 * @return		the maximum Transport agent count
 */
extern int get_agent_Transport_MAX_count();



/** get_agent_Transport_trdefault_count
 * Gets the agent count for the Transport agent type in state trdefault
 * @return		the current Transport agent count in state trdefault
 */
extern int get_agent_Transport_trdefault_count();

/** reset_trdefault_count
 * Resets the agent count of the Transport in state trdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Transport_trdefault_count();

/** get_device_Transport_trdefault_agents
 * Gets a pointer to xmachine_memory_Transport_list on the GPU device
 * @return		a xmachine_memory_Transport_list on the GPU device
 */
extern xmachine_memory_Transport_list* get_device_Transport_trdefault_agents();

/** get_host_Transport_trdefault_agents
 * Gets a pointer to xmachine_memory_Transport_list on the CPU host
 * @return		a xmachine_memory_Transport_list on the CPU host
 */
extern xmachine_memory_Transport_list* get_host_Transport_trdefault_agents();


/** sort_Transports_trdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Transports_trdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Transport_list* agents));


    
/** get_agent_TransportMembership_MAX_count
 * Gets the max agent count for the TransportMembership agent type 
 * @return		the maximum TransportMembership agent count
 */
extern int get_agent_TransportMembership_MAX_count();



/** get_agent_TransportMembership_trmembershipdefault_count
 * Gets the agent count for the TransportMembership agent type in state trmembershipdefault
 * @return		the current TransportMembership agent count in state trmembershipdefault
 */
extern int get_agent_TransportMembership_trmembershipdefault_count();

/** reset_trmembershipdefault_count
 * Resets the agent count of the TransportMembership in state trmembershipdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_TransportMembership_trmembershipdefault_count();

/** get_device_TransportMembership_trmembershipdefault_agents
 * Gets a pointer to xmachine_memory_TransportMembership_list on the GPU device
 * @return		a xmachine_memory_TransportMembership_list on the GPU device
 */
extern xmachine_memory_TransportMembership_list* get_device_TransportMembership_trmembershipdefault_agents();

/** get_host_TransportMembership_trmembershipdefault_agents
 * Gets a pointer to xmachine_memory_TransportMembership_list on the CPU host
 * @return		a xmachine_memory_TransportMembership_list on the CPU host
 */
extern xmachine_memory_TransportMembership_list* get_host_TransportMembership_trmembershipdefault_agents();


/** sort_TransportMemberships_trmembershipdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_TransportMemberships_trmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TransportMembership_list* agents));


    
/** get_agent_Clinic_MAX_count
 * Gets the max agent count for the Clinic agent type 
 * @return		the maximum Clinic agent count
 */
extern int get_agent_Clinic_MAX_count();



/** get_agent_Clinic_cldefault_count
 * Gets the agent count for the Clinic agent type in state cldefault
 * @return		the current Clinic agent count in state cldefault
 */
extern int get_agent_Clinic_cldefault_count();

/** reset_cldefault_count
 * Resets the agent count of the Clinic in state cldefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Clinic_cldefault_count();

/** get_device_Clinic_cldefault_agents
 * Gets a pointer to xmachine_memory_Clinic_list on the GPU device
 * @return		a xmachine_memory_Clinic_list on the GPU device
 */
extern xmachine_memory_Clinic_list* get_device_Clinic_cldefault_agents();

/** get_host_Clinic_cldefault_agents
 * Gets a pointer to xmachine_memory_Clinic_list on the CPU host
 * @return		a xmachine_memory_Clinic_list on the CPU host
 */
extern xmachine_memory_Clinic_list* get_host_Clinic_cldefault_agents();


/** sort_Clinics_cldefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Clinics_cldefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Clinic_list* agents));


    
/** get_agent_Workplace_MAX_count
 * Gets the max agent count for the Workplace agent type 
 * @return		the maximum Workplace agent count
 */
extern int get_agent_Workplace_MAX_count();



/** get_agent_Workplace_wpdefault_count
 * Gets the agent count for the Workplace agent type in state wpdefault
 * @return		the current Workplace agent count in state wpdefault
 */
extern int get_agent_Workplace_wpdefault_count();

/** reset_wpdefault_count
 * Resets the agent count of the Workplace in state wpdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Workplace_wpdefault_count();

/** get_device_Workplace_wpdefault_agents
 * Gets a pointer to xmachine_memory_Workplace_list on the GPU device
 * @return		a xmachine_memory_Workplace_list on the GPU device
 */
extern xmachine_memory_Workplace_list* get_device_Workplace_wpdefault_agents();

/** get_host_Workplace_wpdefault_agents
 * Gets a pointer to xmachine_memory_Workplace_list on the CPU host
 * @return		a xmachine_memory_Workplace_list on the CPU host
 */
extern xmachine_memory_Workplace_list* get_host_Workplace_wpdefault_agents();


/** sort_Workplaces_wpdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Workplaces_wpdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Workplace_list* agents));


    
/** get_agent_WorkplaceMembership_MAX_count
 * Gets the max agent count for the WorkplaceMembership agent type 
 * @return		the maximum WorkplaceMembership agent count
 */
extern int get_agent_WorkplaceMembership_MAX_count();



/** get_agent_WorkplaceMembership_wpmembershipdefault_count
 * Gets the agent count for the WorkplaceMembership agent type in state wpmembershipdefault
 * @return		the current WorkplaceMembership agent count in state wpmembershipdefault
 */
extern int get_agent_WorkplaceMembership_wpmembershipdefault_count();

/** reset_wpmembershipdefault_count
 * Resets the agent count of the WorkplaceMembership in state wpmembershipdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_WorkplaceMembership_wpmembershipdefault_count();

/** get_device_WorkplaceMembership_wpmembershipdefault_agents
 * Gets a pointer to xmachine_memory_WorkplaceMembership_list on the GPU device
 * @return		a xmachine_memory_WorkplaceMembership_list on the GPU device
 */
extern xmachine_memory_WorkplaceMembership_list* get_device_WorkplaceMembership_wpmembershipdefault_agents();

/** get_host_WorkplaceMembership_wpmembershipdefault_agents
 * Gets a pointer to xmachine_memory_WorkplaceMembership_list on the CPU host
 * @return		a xmachine_memory_WorkplaceMembership_list on the CPU host
 */
extern xmachine_memory_WorkplaceMembership_list* get_host_WorkplaceMembership_wpmembershipdefault_agents();


/** sort_WorkplaceMemberships_wpmembershipdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_WorkplaceMemberships_wpmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_WorkplaceMembership_list* agents));


    
/** get_agent_Bar_MAX_count
 * Gets the max agent count for the Bar agent type 
 * @return		the maximum Bar agent count
 */
extern int get_agent_Bar_MAX_count();



/** get_agent_Bar_bdefault_count
 * Gets the agent count for the Bar agent type in state bdefault
 * @return		the current Bar agent count in state bdefault
 */
extern int get_agent_Bar_bdefault_count();

/** reset_bdefault_count
 * Resets the agent count of the Bar in state bdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Bar_bdefault_count();

/** get_device_Bar_bdefault_agents
 * Gets a pointer to xmachine_memory_Bar_list on the GPU device
 * @return		a xmachine_memory_Bar_list on the GPU device
 */
extern xmachine_memory_Bar_list* get_device_Bar_bdefault_agents();

/** get_host_Bar_bdefault_agents
 * Gets a pointer to xmachine_memory_Bar_list on the CPU host
 * @return		a xmachine_memory_Bar_list on the CPU host
 */
extern xmachine_memory_Bar_list* get_host_Bar_bdefault_agents();


/** sort_Bars_bdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Bars_bdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Bar_list* agents));


    
/** get_agent_School_MAX_count
 * Gets the max agent count for the School agent type 
 * @return		the maximum School agent count
 */
extern int get_agent_School_MAX_count();



/** get_agent_School_schdefault_count
 * Gets the agent count for the School agent type in state schdefault
 * @return		the current School agent count in state schdefault
 */
extern int get_agent_School_schdefault_count();

/** reset_schdefault_count
 * Resets the agent count of the School in state schdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_School_schdefault_count();

/** get_device_School_schdefault_agents
 * Gets a pointer to xmachine_memory_School_list on the GPU device
 * @return		a xmachine_memory_School_list on the GPU device
 */
extern xmachine_memory_School_list* get_device_School_schdefault_agents();

/** get_host_School_schdefault_agents
 * Gets a pointer to xmachine_memory_School_list on the CPU host
 * @return		a xmachine_memory_School_list on the CPU host
 */
extern xmachine_memory_School_list* get_host_School_schdefault_agents();


/** sort_Schools_schdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Schools_schdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_School_list* agents));


    
/** get_agent_SchoolMembership_MAX_count
 * Gets the max agent count for the SchoolMembership agent type 
 * @return		the maximum SchoolMembership agent count
 */
extern int get_agent_SchoolMembership_MAX_count();



/** get_agent_SchoolMembership_schmembershipdefault_count
 * Gets the agent count for the SchoolMembership agent type in state schmembershipdefault
 * @return		the current SchoolMembership agent count in state schmembershipdefault
 */
extern int get_agent_SchoolMembership_schmembershipdefault_count();

/** reset_schmembershipdefault_count
 * Resets the agent count of the SchoolMembership in state schmembershipdefault to 0. This is useful for interacting with some visualisations.
 */
extern void reset_SchoolMembership_schmembershipdefault_count();

/** get_device_SchoolMembership_schmembershipdefault_agents
 * Gets a pointer to xmachine_memory_SchoolMembership_list on the GPU device
 * @return		a xmachine_memory_SchoolMembership_list on the GPU device
 */
extern xmachine_memory_SchoolMembership_list* get_device_SchoolMembership_schmembershipdefault_agents();

/** get_host_SchoolMembership_schmembershipdefault_agents
 * Gets a pointer to xmachine_memory_SchoolMembership_list on the CPU host
 * @return		a xmachine_memory_SchoolMembership_list on the CPU host
 */
extern xmachine_memory_SchoolMembership_list* get_host_SchoolMembership_schmembershipdefault_agents();


/** sort_SchoolMemberships_schmembershipdefault
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_SchoolMemberships_schmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_SchoolMembership_list* agents));



/* Host based access of agent variables*/

/** unsigned int get_Person_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_default_variable_id(unsigned int index);

/** unsigned int get_Person_default_variable_step(unsigned int index)
 * Gets the value of the step variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Person_default_variable_step(unsigned int index);

/** unsigned int get_Person_default_variable_householdtime(unsigned int index)
 * Gets the value of the householdtime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdtime
 */
__host__ unsigned int get_Person_default_variable_householdtime(unsigned int index);

/** unsigned int get_Person_default_variable_churchtime(unsigned int index)
 * Gets the value of the churchtime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchtime
 */
__host__ unsigned int get_Person_default_variable_churchtime(unsigned int index);

/** unsigned int get_Person_default_variable_transporttime(unsigned int index)
 * Gets the value of the transporttime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transporttime
 */
__host__ unsigned int get_Person_default_variable_transporttime(unsigned int index);

/** unsigned int get_Person_default_variable_clinictime(unsigned int index)
 * Gets the value of the clinictime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable clinictime
 */
__host__ unsigned int get_Person_default_variable_clinictime(unsigned int index);

/** unsigned int get_Person_default_variable_workplacetime(unsigned int index)
 * Gets the value of the workplacetime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplacetime
 */
__host__ unsigned int get_Person_default_variable_workplacetime(unsigned int index);

/** unsigned int get_Person_default_variable_bartime(unsigned int index)
 * Gets the value of the bartime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable bartime
 */
__host__ unsigned int get_Person_default_variable_bartime(unsigned int index);

/** unsigned int get_Person_default_variable_outsidetime(unsigned int index)
 * Gets the value of the outsidetime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable outsidetime
 */
__host__ unsigned int get_Person_default_variable_outsidetime(unsigned int index);

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

/** unsigned int get_Person_default_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Person_default_variable_churchfreq(unsigned int index);

/** float get_Person_default_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_Person_default_variable_churchdur(unsigned int index);

/** unsigned int get_Person_default_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ unsigned int get_Person_default_variable_transportdur(unsigned int index);

/** int get_Person_default_variable_transportday1(unsigned int index)
 * Gets the value of the transportday1 variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday1
 */
__host__ int get_Person_default_variable_transportday1(unsigned int index);

/** int get_Person_default_variable_transportday2(unsigned int index)
 * Gets the value of the transportday2 variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday2
 */
__host__ int get_Person_default_variable_transportday2(unsigned int index);

/** unsigned int get_Person_default_variable_household(unsigned int index)
 * Gets the value of the household variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household
 */
__host__ unsigned int get_Person_default_variable_household(unsigned int index);

/** int get_Person_default_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ int get_Person_default_variable_church(unsigned int index);

/** int get_Person_default_variable_transport(unsigned int index)
 * Gets the value of the transport variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport
 */
__host__ int get_Person_default_variable_transport(unsigned int index);

/** int get_Person_default_variable_workplace(unsigned int index)
 * Gets the value of the workplace variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace
 */
__host__ int get_Person_default_variable_workplace(unsigned int index);

/** int get_Person_default_variable_school(unsigned int index)
 * Gets the value of the school variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable school
 */
__host__ int get_Person_default_variable_school(unsigned int index);

/** unsigned int get_Person_default_variable_busy(unsigned int index)
 * Gets the value of the busy variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable busy
 */
__host__ unsigned int get_Person_default_variable_busy(unsigned int index);

/** unsigned int get_Person_default_variable_startstep(unsigned int index)
 * Gets the value of the startstep variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable startstep
 */
__host__ unsigned int get_Person_default_variable_startstep(unsigned int index);

/** unsigned int get_Person_default_variable_location(unsigned int index)
 * Gets the value of the location variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location
 */
__host__ unsigned int get_Person_default_variable_location(unsigned int index);

/** unsigned int get_Person_default_variable_locationid(unsigned int index)
 * Gets the value of the locationid variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable locationid
 */
__host__ unsigned int get_Person_default_variable_locationid(unsigned int index);

/** unsigned int get_Person_default_variable_hiv(unsigned int index)
 * Gets the value of the hiv variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hiv
 */
__host__ unsigned int get_Person_default_variable_hiv(unsigned int index);

/** unsigned int get_Person_default_variable_art(unsigned int index)
 * Gets the value of the art variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable art
 */
__host__ unsigned int get_Person_default_variable_art(unsigned int index);

/** unsigned int get_Person_default_variable_activetb(unsigned int index)
 * Gets the value of the activetb variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable activetb
 */
__host__ unsigned int get_Person_default_variable_activetb(unsigned int index);

/** unsigned int get_Person_default_variable_artday(unsigned int index)
 * Gets the value of the artday variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable artday
 */
__host__ unsigned int get_Person_default_variable_artday(unsigned int index);

/** float get_Person_default_variable_p(unsigned int index)
 * Gets the value of the p variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable p
 */
__host__ float get_Person_default_variable_p(unsigned int index);

/** float get_Person_default_variable_q(unsigned int index)
 * Gets the value of the q variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable q
 */
__host__ float get_Person_default_variable_q(unsigned int index);

/** unsigned int get_Person_default_variable_infections(unsigned int index)
 * Gets the value of the infections variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable infections
 */
__host__ unsigned int get_Person_default_variable_infections(unsigned int index);

/** int get_Person_default_variable_lastinfected(unsigned int index)
 * Gets the value of the lastinfected variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfected
 */
__host__ int get_Person_default_variable_lastinfected(unsigned int index);

/** int get_Person_default_variable_lastinfectedid(unsigned int index)
 * Gets the value of the lastinfectedid variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedid
 */
__host__ int get_Person_default_variable_lastinfectedid(unsigned int index);

/** int get_Person_default_variable_lastinfectedtime(unsigned int index)
 * Gets the value of the lastinfectedtime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedtime
 */
__host__ int get_Person_default_variable_lastinfectedtime(unsigned int index);

/** float get_Person_default_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Person_default_variable_lambda(unsigned int index);

/** unsigned int get_Person_default_variable_timevisiting(unsigned int index)
 * Gets the value of the timevisiting variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timevisiting
 */
__host__ unsigned int get_Person_default_variable_timevisiting(unsigned int index);

/** unsigned int get_Person_default_variable_bargoing(unsigned int index)
 * Gets the value of the bargoing variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable bargoing
 */
__host__ unsigned int get_Person_default_variable_bargoing(unsigned int index);

/** unsigned int get_Person_default_variable_barday(unsigned int index)
 * Gets the value of the barday variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable barday
 */
__host__ unsigned int get_Person_default_variable_barday(unsigned int index);

/** unsigned int get_Person_default_variable_schooltime(unsigned int index)
 * Gets the value of the schooltime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable schooltime
 */
__host__ unsigned int get_Person_default_variable_schooltime(unsigned int index);

/** unsigned int get_Person_s2_variable_id(unsigned int index)
 * Gets the value of the id variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Person_s2_variable_id(unsigned int index);

/** unsigned int get_Person_s2_variable_step(unsigned int index)
 * Gets the value of the step variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable step
 */
__host__ unsigned int get_Person_s2_variable_step(unsigned int index);

/** unsigned int get_Person_s2_variable_householdtime(unsigned int index)
 * Gets the value of the householdtime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdtime
 */
__host__ unsigned int get_Person_s2_variable_householdtime(unsigned int index);

/** unsigned int get_Person_s2_variable_churchtime(unsigned int index)
 * Gets the value of the churchtime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchtime
 */
__host__ unsigned int get_Person_s2_variable_churchtime(unsigned int index);

/** unsigned int get_Person_s2_variable_transporttime(unsigned int index)
 * Gets the value of the transporttime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transporttime
 */
__host__ unsigned int get_Person_s2_variable_transporttime(unsigned int index);

/** unsigned int get_Person_s2_variable_clinictime(unsigned int index)
 * Gets the value of the clinictime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable clinictime
 */
__host__ unsigned int get_Person_s2_variable_clinictime(unsigned int index);

/** unsigned int get_Person_s2_variable_workplacetime(unsigned int index)
 * Gets the value of the workplacetime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplacetime
 */
__host__ unsigned int get_Person_s2_variable_workplacetime(unsigned int index);

/** unsigned int get_Person_s2_variable_bartime(unsigned int index)
 * Gets the value of the bartime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable bartime
 */
__host__ unsigned int get_Person_s2_variable_bartime(unsigned int index);

/** unsigned int get_Person_s2_variable_outsidetime(unsigned int index)
 * Gets the value of the outsidetime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable outsidetime
 */
__host__ unsigned int get_Person_s2_variable_outsidetime(unsigned int index);

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

/** unsigned int get_Person_s2_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_Person_s2_variable_churchfreq(unsigned int index);

/** float get_Person_s2_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_Person_s2_variable_churchdur(unsigned int index);

/** unsigned int get_Person_s2_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ unsigned int get_Person_s2_variable_transportdur(unsigned int index);

/** int get_Person_s2_variable_transportday1(unsigned int index)
 * Gets the value of the transportday1 variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday1
 */
__host__ int get_Person_s2_variable_transportday1(unsigned int index);

/** int get_Person_s2_variable_transportday2(unsigned int index)
 * Gets the value of the transportday2 variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday2
 */
__host__ int get_Person_s2_variable_transportday2(unsigned int index);

/** unsigned int get_Person_s2_variable_household(unsigned int index)
 * Gets the value of the household variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household
 */
__host__ unsigned int get_Person_s2_variable_household(unsigned int index);

/** int get_Person_s2_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ int get_Person_s2_variable_church(unsigned int index);

/** int get_Person_s2_variable_transport(unsigned int index)
 * Gets the value of the transport variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport
 */
__host__ int get_Person_s2_variable_transport(unsigned int index);

/** int get_Person_s2_variable_workplace(unsigned int index)
 * Gets the value of the workplace variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace
 */
__host__ int get_Person_s2_variable_workplace(unsigned int index);

/** int get_Person_s2_variable_school(unsigned int index)
 * Gets the value of the school variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable school
 */
__host__ int get_Person_s2_variable_school(unsigned int index);

/** unsigned int get_Person_s2_variable_busy(unsigned int index)
 * Gets the value of the busy variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable busy
 */
__host__ unsigned int get_Person_s2_variable_busy(unsigned int index);

/** unsigned int get_Person_s2_variable_startstep(unsigned int index)
 * Gets the value of the startstep variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable startstep
 */
__host__ unsigned int get_Person_s2_variable_startstep(unsigned int index);

/** unsigned int get_Person_s2_variable_location(unsigned int index)
 * Gets the value of the location variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location
 */
__host__ unsigned int get_Person_s2_variable_location(unsigned int index);

/** unsigned int get_Person_s2_variable_locationid(unsigned int index)
 * Gets the value of the locationid variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable locationid
 */
__host__ unsigned int get_Person_s2_variable_locationid(unsigned int index);

/** unsigned int get_Person_s2_variable_hiv(unsigned int index)
 * Gets the value of the hiv variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hiv
 */
__host__ unsigned int get_Person_s2_variable_hiv(unsigned int index);

/** unsigned int get_Person_s2_variable_art(unsigned int index)
 * Gets the value of the art variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable art
 */
__host__ unsigned int get_Person_s2_variable_art(unsigned int index);

/** unsigned int get_Person_s2_variable_activetb(unsigned int index)
 * Gets the value of the activetb variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable activetb
 */
__host__ unsigned int get_Person_s2_variable_activetb(unsigned int index);

/** unsigned int get_Person_s2_variable_artday(unsigned int index)
 * Gets the value of the artday variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable artday
 */
__host__ unsigned int get_Person_s2_variable_artday(unsigned int index);

/** float get_Person_s2_variable_p(unsigned int index)
 * Gets the value of the p variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable p
 */
__host__ float get_Person_s2_variable_p(unsigned int index);

/** float get_Person_s2_variable_q(unsigned int index)
 * Gets the value of the q variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable q
 */
__host__ float get_Person_s2_variable_q(unsigned int index);

/** unsigned int get_Person_s2_variable_infections(unsigned int index)
 * Gets the value of the infections variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable infections
 */
__host__ unsigned int get_Person_s2_variable_infections(unsigned int index);

/** int get_Person_s2_variable_lastinfected(unsigned int index)
 * Gets the value of the lastinfected variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfected
 */
__host__ int get_Person_s2_variable_lastinfected(unsigned int index);

/** int get_Person_s2_variable_lastinfectedid(unsigned int index)
 * Gets the value of the lastinfectedid variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedid
 */
__host__ int get_Person_s2_variable_lastinfectedid(unsigned int index);

/** int get_Person_s2_variable_lastinfectedtime(unsigned int index)
 * Gets the value of the lastinfectedtime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedtime
 */
__host__ int get_Person_s2_variable_lastinfectedtime(unsigned int index);

/** float get_Person_s2_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Person_s2_variable_lambda(unsigned int index);

/** unsigned int get_Person_s2_variable_timevisiting(unsigned int index)
 * Gets the value of the timevisiting variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timevisiting
 */
__host__ unsigned int get_Person_s2_variable_timevisiting(unsigned int index);

/** unsigned int get_Person_s2_variable_bargoing(unsigned int index)
 * Gets the value of the bargoing variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable bargoing
 */
__host__ unsigned int get_Person_s2_variable_bargoing(unsigned int index);

/** unsigned int get_Person_s2_variable_barday(unsigned int index)
 * Gets the value of the barday variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable barday
 */
__host__ unsigned int get_Person_s2_variable_barday(unsigned int index);

/** unsigned int get_Person_s2_variable_schooltime(unsigned int index)
 * Gets the value of the schooltime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable schooltime
 */
__host__ unsigned int get_Person_s2_variable_schooltime(unsigned int index);

/** unsigned int get_TBAssignment_tbdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an TBAssignment agent in the tbdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_TBAssignment_tbdefault_variable_id(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Household_hhdefault_variable_id(unsigned int index);

/** float get_Household_hhdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Household_hhdefault_variable_lambda(unsigned int index);

/** unsigned int get_Household_hhdefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Household_hhdefault_variable_active(unsigned int index);

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_id(unsigned int index)
 * Gets the value of the household_id variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_id
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_id(unsigned int index);

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_person_id(unsigned int index);

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_size(unsigned int index)
 * Gets the value of the household_size variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_size
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_size(unsigned int index);

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchgoing(unsigned int index)
 * Gets the value of the churchgoing variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchgoing
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchgoing(unsigned int index);

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchfreq(unsigned int index)
 * Gets the value of the churchfreq variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchfreq
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_churchfreq(unsigned int index);

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

/** float get_Church_chudefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Church_chudefault_variable_lambda(unsigned int index);

/** unsigned int get_Church_chudefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Church_chudefault_variable_active(unsigned int index);

/** unsigned int get_ChurchMembership_chumembershipdefault_variable_church_id(unsigned int index)
 * Gets the value of the church_id variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church_id
 */
__host__ unsigned int get_ChurchMembership_chumembershipdefault_variable_church_id(unsigned int index);

/** unsigned int get_ChurchMembership_chumembershipdefault_variable_household_id(unsigned int index)
 * Gets the value of the household_id variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_id
 */
__host__ unsigned int get_ChurchMembership_chumembershipdefault_variable_household_id(unsigned int index);

/** float get_ChurchMembership_chumembershipdefault_variable_churchdur(unsigned int index)
 * Gets the value of the churchdur variable of an ChurchMembership agent in the chumembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchdur
 */
__host__ float get_ChurchMembership_chumembershipdefault_variable_churchdur(unsigned int index);

/** unsigned int get_Transport_trdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Transport_trdefault_variable_id(unsigned int index);

/** float get_Transport_trdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Transport_trdefault_variable_lambda(unsigned int index);

/** unsigned int get_Transport_trdefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Transport_trdefault_variable_active(unsigned int index);

/** int get_TransportMembership_trmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ int get_TransportMembership_trmembershipdefault_variable_person_id(unsigned int index);

/** unsigned int get_TransportMembership_trmembershipdefault_variable_transport_id(unsigned int index)
 * Gets the value of the transport_id variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport_id
 */
__host__ unsigned int get_TransportMembership_trmembershipdefault_variable_transport_id(unsigned int index);

/** unsigned int get_TransportMembership_trmembershipdefault_variable_duration(unsigned int index)
 * Gets the value of the duration variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable duration
 */
__host__ unsigned int get_TransportMembership_trmembershipdefault_variable_duration(unsigned int index);

/** unsigned int get_Clinic_cldefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Clinic agent in the cldefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Clinic_cldefault_variable_id(unsigned int index);

/** float get_Clinic_cldefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Clinic agent in the cldefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Clinic_cldefault_variable_lambda(unsigned int index);

/** unsigned int get_Workplace_wpdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Workplace agent in the wpdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Workplace_wpdefault_variable_id(unsigned int index);

/** float get_Workplace_wpdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Workplace agent in the wpdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Workplace_wpdefault_variable_lambda(unsigned int index);

/** unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an WorkplaceMembership agent in the wpmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_person_id(unsigned int index);

/** unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_workplace_id(unsigned int index)
 * Gets the value of the workplace_id variable of an WorkplaceMembership agent in the wpmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace_id
 */
__host__ unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_workplace_id(unsigned int index);

/** unsigned int get_Bar_bdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Bar agent in the bdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Bar_bdefault_variable_id(unsigned int index);

/** float get_Bar_bdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Bar agent in the bdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Bar_bdefault_variable_lambda(unsigned int index);

/** unsigned int get_School_schdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an School agent in the schdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_School_schdefault_variable_id(unsigned int index);

/** float get_School_schdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an School agent in the schdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_School_schdefault_variable_lambda(unsigned int index);

/** unsigned int get_SchoolMembership_schmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an SchoolMembership agent in the schmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ unsigned int get_SchoolMembership_schmembershipdefault_variable_person_id(unsigned int index);

/** unsigned int get_SchoolMembership_schmembershipdefault_variable_school_id(unsigned int index)
 * Gets the value of the school_id variable of an SchoolMembership agent in the schmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable school_id
 */
__host__ unsigned int get_SchoolMembership_schmembershipdefault_variable_school_id(unsigned int index);




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

/** h_allocate_agent_TBAssignment
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated TBAssignment struct.
 */
xmachine_memory_TBAssignment* h_allocate_agent_TBAssignment();
/** h_free_agent_TBAssignment
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_TBAssignment(xmachine_memory_TBAssignment** agent);
/** h_allocate_agent_TBAssignment_array
 * Utility function to allocate an array of structs for  TBAssignment agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_TBAssignment** h_allocate_agent_TBAssignment_array(unsigned int count);
/** h_free_agent_TBAssignment_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_TBAssignment_array(xmachine_memory_TBAssignment*** agents, unsigned int count);


/** h_add_agent_TBAssignment_tbdefault
 * Host function to add a single agent of type TBAssignment to the tbdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_TBAssignment_tbdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_TBAssignment_tbdefault(xmachine_memory_TBAssignment* agent);

/** h_add_agents_TBAssignment_tbdefault(
 * Host function to add multiple agents of type TBAssignment to the tbdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of TBAssignment agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_TBAssignment_tbdefault(xmachine_memory_TBAssignment** agents, unsigned int count);

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

/** h_allocate_agent_HouseholdMembership
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated HouseholdMembership struct.
 */
xmachine_memory_HouseholdMembership* h_allocate_agent_HouseholdMembership();
/** h_free_agent_HouseholdMembership
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_HouseholdMembership(xmachine_memory_HouseholdMembership** agent);
/** h_allocate_agent_HouseholdMembership_array
 * Utility function to allocate an array of structs for  HouseholdMembership agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_HouseholdMembership** h_allocate_agent_HouseholdMembership_array(unsigned int count);
/** h_free_agent_HouseholdMembership_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_HouseholdMembership_array(xmachine_memory_HouseholdMembership*** agents, unsigned int count);


/** h_add_agent_HouseholdMembership_hhmembershipdefault
 * Host function to add a single agent of type HouseholdMembership to the hhmembershipdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_HouseholdMembership_hhmembershipdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_HouseholdMembership_hhmembershipdefault(xmachine_memory_HouseholdMembership* agent);

/** h_add_agents_HouseholdMembership_hhmembershipdefault(
 * Host function to add multiple agents of type HouseholdMembership to the hhmembershipdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of HouseholdMembership agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_HouseholdMembership_hhmembershipdefault(xmachine_memory_HouseholdMembership** agents, unsigned int count);

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

/** h_allocate_agent_ChurchMembership
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated ChurchMembership struct.
 */
xmachine_memory_ChurchMembership* h_allocate_agent_ChurchMembership();
/** h_free_agent_ChurchMembership
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_ChurchMembership(xmachine_memory_ChurchMembership** agent);
/** h_allocate_agent_ChurchMembership_array
 * Utility function to allocate an array of structs for  ChurchMembership agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_ChurchMembership** h_allocate_agent_ChurchMembership_array(unsigned int count);
/** h_free_agent_ChurchMembership_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_ChurchMembership_array(xmachine_memory_ChurchMembership*** agents, unsigned int count);


/** h_add_agent_ChurchMembership_chumembershipdefault
 * Host function to add a single agent of type ChurchMembership to the chumembershipdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_ChurchMembership_chumembershipdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_ChurchMembership_chumembershipdefault(xmachine_memory_ChurchMembership* agent);

/** h_add_agents_ChurchMembership_chumembershipdefault(
 * Host function to add multiple agents of type ChurchMembership to the chumembershipdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of ChurchMembership agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_ChurchMembership_chumembershipdefault(xmachine_memory_ChurchMembership** agents, unsigned int count);

/** h_allocate_agent_Transport
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Transport struct.
 */
xmachine_memory_Transport* h_allocate_agent_Transport();
/** h_free_agent_Transport
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Transport(xmachine_memory_Transport** agent);
/** h_allocate_agent_Transport_array
 * Utility function to allocate an array of structs for  Transport agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Transport** h_allocate_agent_Transport_array(unsigned int count);
/** h_free_agent_Transport_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Transport_array(xmachine_memory_Transport*** agents, unsigned int count);


/** h_add_agent_Transport_trdefault
 * Host function to add a single agent of type Transport to the trdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Transport_trdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Transport_trdefault(xmachine_memory_Transport* agent);

/** h_add_agents_Transport_trdefault(
 * Host function to add multiple agents of type Transport to the trdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Transport agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Transport_trdefault(xmachine_memory_Transport** agents, unsigned int count);

/** h_allocate_agent_TransportMembership
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated TransportMembership struct.
 */
xmachine_memory_TransportMembership* h_allocate_agent_TransportMembership();
/** h_free_agent_TransportMembership
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_TransportMembership(xmachine_memory_TransportMembership** agent);
/** h_allocate_agent_TransportMembership_array
 * Utility function to allocate an array of structs for  TransportMembership agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_TransportMembership** h_allocate_agent_TransportMembership_array(unsigned int count);
/** h_free_agent_TransportMembership_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_TransportMembership_array(xmachine_memory_TransportMembership*** agents, unsigned int count);


/** h_add_agent_TransportMembership_trmembershipdefault
 * Host function to add a single agent of type TransportMembership to the trmembershipdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_TransportMembership_trmembershipdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_TransportMembership_trmembershipdefault(xmachine_memory_TransportMembership* agent);

/** h_add_agents_TransportMembership_trmembershipdefault(
 * Host function to add multiple agents of type TransportMembership to the trmembershipdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of TransportMembership agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_TransportMembership_trmembershipdefault(xmachine_memory_TransportMembership** agents, unsigned int count);

/** h_allocate_agent_Clinic
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Clinic struct.
 */
xmachine_memory_Clinic* h_allocate_agent_Clinic();
/** h_free_agent_Clinic
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Clinic(xmachine_memory_Clinic** agent);
/** h_allocate_agent_Clinic_array
 * Utility function to allocate an array of structs for  Clinic agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Clinic** h_allocate_agent_Clinic_array(unsigned int count);
/** h_free_agent_Clinic_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Clinic_array(xmachine_memory_Clinic*** agents, unsigned int count);


/** h_add_agent_Clinic_cldefault
 * Host function to add a single agent of type Clinic to the cldefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Clinic_cldefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Clinic_cldefault(xmachine_memory_Clinic* agent);

/** h_add_agents_Clinic_cldefault(
 * Host function to add multiple agents of type Clinic to the cldefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Clinic agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Clinic_cldefault(xmachine_memory_Clinic** agents, unsigned int count);

/** h_allocate_agent_Workplace
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Workplace struct.
 */
xmachine_memory_Workplace* h_allocate_agent_Workplace();
/** h_free_agent_Workplace
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Workplace(xmachine_memory_Workplace** agent);
/** h_allocate_agent_Workplace_array
 * Utility function to allocate an array of structs for  Workplace agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Workplace** h_allocate_agent_Workplace_array(unsigned int count);
/** h_free_agent_Workplace_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Workplace_array(xmachine_memory_Workplace*** agents, unsigned int count);


/** h_add_agent_Workplace_wpdefault
 * Host function to add a single agent of type Workplace to the wpdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Workplace_wpdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Workplace_wpdefault(xmachine_memory_Workplace* agent);

/** h_add_agents_Workplace_wpdefault(
 * Host function to add multiple agents of type Workplace to the wpdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Workplace agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Workplace_wpdefault(xmachine_memory_Workplace** agents, unsigned int count);

/** h_allocate_agent_WorkplaceMembership
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated WorkplaceMembership struct.
 */
xmachine_memory_WorkplaceMembership* h_allocate_agent_WorkplaceMembership();
/** h_free_agent_WorkplaceMembership
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_WorkplaceMembership(xmachine_memory_WorkplaceMembership** agent);
/** h_allocate_agent_WorkplaceMembership_array
 * Utility function to allocate an array of structs for  WorkplaceMembership agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_WorkplaceMembership** h_allocate_agent_WorkplaceMembership_array(unsigned int count);
/** h_free_agent_WorkplaceMembership_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_WorkplaceMembership_array(xmachine_memory_WorkplaceMembership*** agents, unsigned int count);


/** h_add_agent_WorkplaceMembership_wpmembershipdefault
 * Host function to add a single agent of type WorkplaceMembership to the wpmembershipdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_WorkplaceMembership_wpmembershipdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_WorkplaceMembership_wpmembershipdefault(xmachine_memory_WorkplaceMembership* agent);

/** h_add_agents_WorkplaceMembership_wpmembershipdefault(
 * Host function to add multiple agents of type WorkplaceMembership to the wpmembershipdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of WorkplaceMembership agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_WorkplaceMembership_wpmembershipdefault(xmachine_memory_WorkplaceMembership** agents, unsigned int count);

/** h_allocate_agent_Bar
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Bar struct.
 */
xmachine_memory_Bar* h_allocate_agent_Bar();
/** h_free_agent_Bar
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Bar(xmachine_memory_Bar** agent);
/** h_allocate_agent_Bar_array
 * Utility function to allocate an array of structs for  Bar agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Bar** h_allocate_agent_Bar_array(unsigned int count);
/** h_free_agent_Bar_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Bar_array(xmachine_memory_Bar*** agents, unsigned int count);


/** h_add_agent_Bar_bdefault
 * Host function to add a single agent of type Bar to the bdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Bar_bdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Bar_bdefault(xmachine_memory_Bar* agent);

/** h_add_agents_Bar_bdefault(
 * Host function to add multiple agents of type Bar to the bdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Bar agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Bar_bdefault(xmachine_memory_Bar** agents, unsigned int count);

/** h_allocate_agent_School
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated School struct.
 */
xmachine_memory_School* h_allocate_agent_School();
/** h_free_agent_School
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_School(xmachine_memory_School** agent);
/** h_allocate_agent_School_array
 * Utility function to allocate an array of structs for  School agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_School** h_allocate_agent_School_array(unsigned int count);
/** h_free_agent_School_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_School_array(xmachine_memory_School*** agents, unsigned int count);


/** h_add_agent_School_schdefault
 * Host function to add a single agent of type School to the schdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_School_schdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_School_schdefault(xmachine_memory_School* agent);

/** h_add_agents_School_schdefault(
 * Host function to add multiple agents of type School to the schdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of School agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_School_schdefault(xmachine_memory_School** agents, unsigned int count);

/** h_allocate_agent_SchoolMembership
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated SchoolMembership struct.
 */
xmachine_memory_SchoolMembership* h_allocate_agent_SchoolMembership();
/** h_free_agent_SchoolMembership
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_SchoolMembership(xmachine_memory_SchoolMembership** agent);
/** h_allocate_agent_SchoolMembership_array
 * Utility function to allocate an array of structs for  SchoolMembership agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_SchoolMembership** h_allocate_agent_SchoolMembership_array(unsigned int count);
/** h_free_agent_SchoolMembership_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_SchoolMembership_array(xmachine_memory_SchoolMembership*** agents, unsigned int count);


/** h_add_agent_SchoolMembership_schmembershipdefault
 * Host function to add a single agent of type SchoolMembership to the schmembershipdefault state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_SchoolMembership_schmembershipdefault instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_SchoolMembership_schmembershipdefault(xmachine_memory_SchoolMembership* agent);

/** h_add_agents_SchoolMembership_schmembershipdefault(
 * Host function to add multiple agents of type SchoolMembership to the schmembershipdefault state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of SchoolMembership agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_SchoolMembership_schmembershipdefault(xmachine_memory_SchoolMembership** agents, unsigned int count);

  
  
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

/** unsigned int reduce_Person_default_step_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_step_variable();



/** unsigned int count_Person_default_step_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_step_variable(int count_value);

/** unsigned int min_Person_default_step_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_step_variable();
/** unsigned int max_Person_default_step_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_step_variable();

/** unsigned int reduce_Person_default_householdtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_householdtime_variable();



/** unsigned int count_Person_default_householdtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_householdtime_variable(int count_value);

/** unsigned int min_Person_default_householdtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_householdtime_variable();
/** unsigned int max_Person_default_householdtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_householdtime_variable();

/** unsigned int reduce_Person_default_churchtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_churchtime_variable();



/** unsigned int count_Person_default_churchtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_churchtime_variable(int count_value);

/** unsigned int min_Person_default_churchtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_churchtime_variable();
/** unsigned int max_Person_default_churchtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_churchtime_variable();

/** unsigned int reduce_Person_default_transporttime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_transporttime_variable();



/** unsigned int count_Person_default_transporttime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_transporttime_variable(int count_value);

/** unsigned int min_Person_default_transporttime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_transporttime_variable();
/** unsigned int max_Person_default_transporttime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_transporttime_variable();

/** unsigned int reduce_Person_default_clinictime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_clinictime_variable();



/** unsigned int count_Person_default_clinictime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_clinictime_variable(int count_value);

/** unsigned int min_Person_default_clinictime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_clinictime_variable();
/** unsigned int max_Person_default_clinictime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_clinictime_variable();

/** unsigned int reduce_Person_default_workplacetime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_workplacetime_variable();



/** unsigned int count_Person_default_workplacetime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_workplacetime_variable(int count_value);

/** unsigned int min_Person_default_workplacetime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_workplacetime_variable();
/** unsigned int max_Person_default_workplacetime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_workplacetime_variable();

/** unsigned int reduce_Person_default_bartime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_bartime_variable();



/** unsigned int count_Person_default_bartime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_bartime_variable(int count_value);

/** unsigned int min_Person_default_bartime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_bartime_variable();
/** unsigned int max_Person_default_bartime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_bartime_variable();

/** unsigned int reduce_Person_default_outsidetime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_outsidetime_variable();



/** unsigned int count_Person_default_outsidetime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_outsidetime_variable(int count_value);

/** unsigned int min_Person_default_outsidetime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_outsidetime_variable();
/** unsigned int max_Person_default_outsidetime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_outsidetime_variable();

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

/** unsigned int reduce_Person_default_churchfreq_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_churchfreq_variable();



/** unsigned int count_Person_default_churchfreq_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_churchfreq_variable(int count_value);

/** unsigned int min_Person_default_churchfreq_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_churchfreq_variable();
/** unsigned int max_Person_default_churchfreq_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_churchfreq_variable();

/** float reduce_Person_default_churchdur_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_default_churchdur_variable();



/** float min_Person_default_churchdur_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_default_churchdur_variable();
/** float max_Person_default_churchdur_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_default_churchdur_variable();

/** unsigned int reduce_Person_default_transportdur_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_transportdur_variable();



/** unsigned int count_Person_default_transportdur_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_transportdur_variable(int count_value);

/** unsigned int min_Person_default_transportdur_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_transportdur_variable();
/** unsigned int max_Person_default_transportdur_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_transportdur_variable();

/** int reduce_Person_default_transportday1_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_transportday1_variable();



/** int count_Person_default_transportday1_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_transportday1_variable(int count_value);

/** int min_Person_default_transportday1_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_transportday1_variable();
/** int max_Person_default_transportday1_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_transportday1_variable();

/** int reduce_Person_default_transportday2_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_transportday2_variable();



/** int count_Person_default_transportday2_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_transportday2_variable(int count_value);

/** int min_Person_default_transportday2_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_transportday2_variable();
/** int max_Person_default_transportday2_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_transportday2_variable();

/** unsigned int reduce_Person_default_household_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_household_variable();



/** unsigned int count_Person_default_household_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_household_variable(int count_value);

/** unsigned int min_Person_default_household_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_household_variable();
/** unsigned int max_Person_default_household_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_household_variable();

/** int reduce_Person_default_church_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_church_variable();



/** int count_Person_default_church_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_church_variable(int count_value);

/** int min_Person_default_church_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_church_variable();
/** int max_Person_default_church_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_church_variable();

/** int reduce_Person_default_transport_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_transport_variable();



/** int count_Person_default_transport_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_transport_variable(int count_value);

/** int min_Person_default_transport_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_transport_variable();
/** int max_Person_default_transport_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_transport_variable();

/** int reduce_Person_default_workplace_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_workplace_variable();



/** int count_Person_default_workplace_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_workplace_variable(int count_value);

/** int min_Person_default_workplace_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_workplace_variable();
/** int max_Person_default_workplace_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_workplace_variable();

/** int reduce_Person_default_school_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_school_variable();



/** int count_Person_default_school_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_school_variable(int count_value);

/** int min_Person_default_school_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_school_variable();
/** int max_Person_default_school_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_school_variable();

/** unsigned int reduce_Person_default_busy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_busy_variable();



/** unsigned int count_Person_default_busy_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_busy_variable(int count_value);

/** unsigned int min_Person_default_busy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_busy_variable();
/** unsigned int max_Person_default_busy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_busy_variable();

/** unsigned int reduce_Person_default_startstep_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_startstep_variable();



/** unsigned int count_Person_default_startstep_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_startstep_variable(int count_value);

/** unsigned int min_Person_default_startstep_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_startstep_variable();
/** unsigned int max_Person_default_startstep_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_startstep_variable();

/** unsigned int reduce_Person_default_location_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_location_variable();



/** unsigned int count_Person_default_location_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_location_variable(int count_value);

/** unsigned int min_Person_default_location_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_location_variable();
/** unsigned int max_Person_default_location_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_location_variable();

/** unsigned int reduce_Person_default_locationid_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_locationid_variable();



/** unsigned int count_Person_default_locationid_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_locationid_variable(int count_value);

/** unsigned int min_Person_default_locationid_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_locationid_variable();
/** unsigned int max_Person_default_locationid_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_locationid_variable();

/** unsigned int reduce_Person_default_hiv_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_hiv_variable();



/** unsigned int count_Person_default_hiv_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_hiv_variable(int count_value);

/** unsigned int min_Person_default_hiv_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_hiv_variable();
/** unsigned int max_Person_default_hiv_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_hiv_variable();

/** unsigned int reduce_Person_default_art_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_art_variable();



/** unsigned int count_Person_default_art_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_art_variable(int count_value);

/** unsigned int min_Person_default_art_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_art_variable();
/** unsigned int max_Person_default_art_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_art_variable();

/** unsigned int reduce_Person_default_activetb_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_activetb_variable();



/** unsigned int count_Person_default_activetb_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_activetb_variable(int count_value);

/** unsigned int min_Person_default_activetb_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_activetb_variable();
/** unsigned int max_Person_default_activetb_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_activetb_variable();

/** unsigned int reduce_Person_default_artday_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_artday_variable();



/** unsigned int count_Person_default_artday_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_artday_variable(int count_value);

/** unsigned int min_Person_default_artday_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_artday_variable();
/** unsigned int max_Person_default_artday_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_artday_variable();

/** float reduce_Person_default_p_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_default_p_variable();



/** float min_Person_default_p_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_default_p_variable();
/** float max_Person_default_p_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_default_p_variable();

/** float reduce_Person_default_q_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_default_q_variable();



/** float min_Person_default_q_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_default_q_variable();
/** float max_Person_default_q_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_default_q_variable();

/** unsigned int reduce_Person_default_infections_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_infections_variable();



/** unsigned int count_Person_default_infections_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_infections_variable(int count_value);

/** unsigned int min_Person_default_infections_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_infections_variable();
/** unsigned int max_Person_default_infections_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_infections_variable();

/** int reduce_Person_default_lastinfected_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_lastinfected_variable();



/** int count_Person_default_lastinfected_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_lastinfected_variable(int count_value);

/** int min_Person_default_lastinfected_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_lastinfected_variable();
/** int max_Person_default_lastinfected_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_lastinfected_variable();

/** int reduce_Person_default_lastinfectedid_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_lastinfectedid_variable();



/** int count_Person_default_lastinfectedid_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_lastinfectedid_variable(int count_value);

/** int min_Person_default_lastinfectedid_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_lastinfectedid_variable();
/** int max_Person_default_lastinfectedid_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_lastinfectedid_variable();

/** int reduce_Person_default_lastinfectedtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_default_lastinfectedtime_variable();



/** int count_Person_default_lastinfectedtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_default_lastinfectedtime_variable(int count_value);

/** int min_Person_default_lastinfectedtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_default_lastinfectedtime_variable();
/** int max_Person_default_lastinfectedtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_default_lastinfectedtime_variable();

/** float reduce_Person_default_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_default_lambda_variable();



/** float min_Person_default_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_default_lambda_variable();
/** float max_Person_default_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_default_lambda_variable();

/** unsigned int reduce_Person_default_timevisiting_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_timevisiting_variable();



/** unsigned int count_Person_default_timevisiting_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_timevisiting_variable(int count_value);

/** unsigned int min_Person_default_timevisiting_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_timevisiting_variable();
/** unsigned int max_Person_default_timevisiting_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_timevisiting_variable();

/** unsigned int reduce_Person_default_bargoing_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_bargoing_variable();



/** unsigned int count_Person_default_bargoing_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_bargoing_variable(int count_value);

/** unsigned int min_Person_default_bargoing_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_bargoing_variable();
/** unsigned int max_Person_default_bargoing_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_bargoing_variable();

/** unsigned int reduce_Person_default_barday_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_barday_variable();



/** unsigned int count_Person_default_barday_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_barday_variable(int count_value);

/** unsigned int min_Person_default_barday_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_barday_variable();
/** unsigned int max_Person_default_barday_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_barday_variable();

/** unsigned int reduce_Person_default_schooltime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_default_schooltime_variable();



/** unsigned int count_Person_default_schooltime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_default_schooltime_variable(int count_value);

/** unsigned int min_Person_default_schooltime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_default_schooltime_variable();
/** unsigned int max_Person_default_schooltime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_default_schooltime_variable();

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

/** unsigned int reduce_Person_s2_step_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_step_variable();



/** unsigned int count_Person_s2_step_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_step_variable(int count_value);

/** unsigned int min_Person_s2_step_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_step_variable();
/** unsigned int max_Person_s2_step_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_step_variable();

/** unsigned int reduce_Person_s2_householdtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_householdtime_variable();



/** unsigned int count_Person_s2_householdtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_householdtime_variable(int count_value);

/** unsigned int min_Person_s2_householdtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_householdtime_variable();
/** unsigned int max_Person_s2_householdtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_householdtime_variable();

/** unsigned int reduce_Person_s2_churchtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_churchtime_variable();



/** unsigned int count_Person_s2_churchtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_churchtime_variable(int count_value);

/** unsigned int min_Person_s2_churchtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_churchtime_variable();
/** unsigned int max_Person_s2_churchtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_churchtime_variable();

/** unsigned int reduce_Person_s2_transporttime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_transporttime_variable();



/** unsigned int count_Person_s2_transporttime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_transporttime_variable(int count_value);

/** unsigned int min_Person_s2_transporttime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_transporttime_variable();
/** unsigned int max_Person_s2_transporttime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_transporttime_variable();

/** unsigned int reduce_Person_s2_clinictime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_clinictime_variable();



/** unsigned int count_Person_s2_clinictime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_clinictime_variable(int count_value);

/** unsigned int min_Person_s2_clinictime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_clinictime_variable();
/** unsigned int max_Person_s2_clinictime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_clinictime_variable();

/** unsigned int reduce_Person_s2_workplacetime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_workplacetime_variable();



/** unsigned int count_Person_s2_workplacetime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_workplacetime_variable(int count_value);

/** unsigned int min_Person_s2_workplacetime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_workplacetime_variable();
/** unsigned int max_Person_s2_workplacetime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_workplacetime_variable();

/** unsigned int reduce_Person_s2_bartime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_bartime_variable();



/** unsigned int count_Person_s2_bartime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_bartime_variable(int count_value);

/** unsigned int min_Person_s2_bartime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_bartime_variable();
/** unsigned int max_Person_s2_bartime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_bartime_variable();

/** unsigned int reduce_Person_s2_outsidetime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_outsidetime_variable();



/** unsigned int count_Person_s2_outsidetime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_outsidetime_variable(int count_value);

/** unsigned int min_Person_s2_outsidetime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_outsidetime_variable();
/** unsigned int max_Person_s2_outsidetime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_outsidetime_variable();

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

/** unsigned int reduce_Person_s2_churchfreq_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_churchfreq_variable();



/** unsigned int count_Person_s2_churchfreq_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_churchfreq_variable(int count_value);

/** unsigned int min_Person_s2_churchfreq_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_churchfreq_variable();
/** unsigned int max_Person_s2_churchfreq_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_churchfreq_variable();

/** float reduce_Person_s2_churchdur_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_s2_churchdur_variable();



/** float min_Person_s2_churchdur_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_s2_churchdur_variable();
/** float max_Person_s2_churchdur_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_s2_churchdur_variable();

/** unsigned int reduce_Person_s2_transportdur_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_transportdur_variable();



/** unsigned int count_Person_s2_transportdur_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_transportdur_variable(int count_value);

/** unsigned int min_Person_s2_transportdur_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_transportdur_variable();
/** unsigned int max_Person_s2_transportdur_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_transportdur_variable();

/** int reduce_Person_s2_transportday1_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_transportday1_variable();



/** int count_Person_s2_transportday1_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_transportday1_variable(int count_value);

/** int min_Person_s2_transportday1_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_transportday1_variable();
/** int max_Person_s2_transportday1_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_transportday1_variable();

/** int reduce_Person_s2_transportday2_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_transportday2_variable();



/** int count_Person_s2_transportday2_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_transportday2_variable(int count_value);

/** int min_Person_s2_transportday2_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_transportday2_variable();
/** int max_Person_s2_transportday2_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_transportday2_variable();

/** unsigned int reduce_Person_s2_household_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_household_variable();



/** unsigned int count_Person_s2_household_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_household_variable(int count_value);

/** unsigned int min_Person_s2_household_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_household_variable();
/** unsigned int max_Person_s2_household_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_household_variable();

/** int reduce_Person_s2_church_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_church_variable();



/** int count_Person_s2_church_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_church_variable(int count_value);

/** int min_Person_s2_church_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_church_variable();
/** int max_Person_s2_church_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_church_variable();

/** int reduce_Person_s2_transport_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_transport_variable();



/** int count_Person_s2_transport_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_transport_variable(int count_value);

/** int min_Person_s2_transport_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_transport_variable();
/** int max_Person_s2_transport_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_transport_variable();

/** int reduce_Person_s2_workplace_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_workplace_variable();



/** int count_Person_s2_workplace_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_workplace_variable(int count_value);

/** int min_Person_s2_workplace_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_workplace_variable();
/** int max_Person_s2_workplace_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_workplace_variable();

/** int reduce_Person_s2_school_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_school_variable();



/** int count_Person_s2_school_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_school_variable(int count_value);

/** int min_Person_s2_school_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_school_variable();
/** int max_Person_s2_school_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_school_variable();

/** unsigned int reduce_Person_s2_busy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_busy_variable();



/** unsigned int count_Person_s2_busy_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_busy_variable(int count_value);

/** unsigned int min_Person_s2_busy_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_busy_variable();
/** unsigned int max_Person_s2_busy_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_busy_variable();

/** unsigned int reduce_Person_s2_startstep_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_startstep_variable();



/** unsigned int count_Person_s2_startstep_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_startstep_variable(int count_value);

/** unsigned int min_Person_s2_startstep_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_startstep_variable();
/** unsigned int max_Person_s2_startstep_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_startstep_variable();

/** unsigned int reduce_Person_s2_location_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_location_variable();



/** unsigned int count_Person_s2_location_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_location_variable(int count_value);

/** unsigned int min_Person_s2_location_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_location_variable();
/** unsigned int max_Person_s2_location_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_location_variable();

/** unsigned int reduce_Person_s2_locationid_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_locationid_variable();



/** unsigned int count_Person_s2_locationid_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_locationid_variable(int count_value);

/** unsigned int min_Person_s2_locationid_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_locationid_variable();
/** unsigned int max_Person_s2_locationid_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_locationid_variable();

/** unsigned int reduce_Person_s2_hiv_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_hiv_variable();



/** unsigned int count_Person_s2_hiv_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_hiv_variable(int count_value);

/** unsigned int min_Person_s2_hiv_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_hiv_variable();
/** unsigned int max_Person_s2_hiv_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_hiv_variable();

/** unsigned int reduce_Person_s2_art_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_art_variable();



/** unsigned int count_Person_s2_art_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_art_variable(int count_value);

/** unsigned int min_Person_s2_art_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_art_variable();
/** unsigned int max_Person_s2_art_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_art_variable();

/** unsigned int reduce_Person_s2_activetb_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_activetb_variable();



/** unsigned int count_Person_s2_activetb_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_activetb_variable(int count_value);

/** unsigned int min_Person_s2_activetb_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_activetb_variable();
/** unsigned int max_Person_s2_activetb_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_activetb_variable();

/** unsigned int reduce_Person_s2_artday_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_artday_variable();



/** unsigned int count_Person_s2_artday_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_artday_variable(int count_value);

/** unsigned int min_Person_s2_artday_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_artday_variable();
/** unsigned int max_Person_s2_artday_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_artday_variable();

/** float reduce_Person_s2_p_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_s2_p_variable();



/** float min_Person_s2_p_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_s2_p_variable();
/** float max_Person_s2_p_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_s2_p_variable();

/** float reduce_Person_s2_q_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_s2_q_variable();



/** float min_Person_s2_q_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_s2_q_variable();
/** float max_Person_s2_q_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_s2_q_variable();

/** unsigned int reduce_Person_s2_infections_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_infections_variable();



/** unsigned int count_Person_s2_infections_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_infections_variable(int count_value);

/** unsigned int min_Person_s2_infections_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_infections_variable();
/** unsigned int max_Person_s2_infections_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_infections_variable();

/** int reduce_Person_s2_lastinfected_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_lastinfected_variable();



/** int count_Person_s2_lastinfected_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_lastinfected_variable(int count_value);

/** int min_Person_s2_lastinfected_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_lastinfected_variable();
/** int max_Person_s2_lastinfected_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_lastinfected_variable();

/** int reduce_Person_s2_lastinfectedid_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_lastinfectedid_variable();



/** int count_Person_s2_lastinfectedid_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_lastinfectedid_variable(int count_value);

/** int min_Person_s2_lastinfectedid_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_lastinfectedid_variable();
/** int max_Person_s2_lastinfectedid_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_lastinfectedid_variable();

/** int reduce_Person_s2_lastinfectedtime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Person_s2_lastinfectedtime_variable();



/** int count_Person_s2_lastinfectedtime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Person_s2_lastinfectedtime_variable(int count_value);

/** int min_Person_s2_lastinfectedtime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Person_s2_lastinfectedtime_variable();
/** int max_Person_s2_lastinfectedtime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Person_s2_lastinfectedtime_variable();

/** float reduce_Person_s2_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Person_s2_lambda_variable();



/** float min_Person_s2_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Person_s2_lambda_variable();
/** float max_Person_s2_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Person_s2_lambda_variable();

/** unsigned int reduce_Person_s2_timevisiting_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_timevisiting_variable();



/** unsigned int count_Person_s2_timevisiting_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_timevisiting_variable(int count_value);

/** unsigned int min_Person_s2_timevisiting_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_timevisiting_variable();
/** unsigned int max_Person_s2_timevisiting_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_timevisiting_variable();

/** unsigned int reduce_Person_s2_bargoing_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_bargoing_variable();



/** unsigned int count_Person_s2_bargoing_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_bargoing_variable(int count_value);

/** unsigned int min_Person_s2_bargoing_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_bargoing_variable();
/** unsigned int max_Person_s2_bargoing_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_bargoing_variable();

/** unsigned int reduce_Person_s2_barday_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_barday_variable();



/** unsigned int count_Person_s2_barday_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_barday_variable(int count_value);

/** unsigned int min_Person_s2_barday_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_barday_variable();
/** unsigned int max_Person_s2_barday_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_barday_variable();

/** unsigned int reduce_Person_s2_schooltime_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Person_s2_schooltime_variable();



/** unsigned int count_Person_s2_schooltime_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Person_s2_schooltime_variable(int count_value);

/** unsigned int min_Person_s2_schooltime_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Person_s2_schooltime_variable();
/** unsigned int max_Person_s2_schooltime_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Person_s2_schooltime_variable();

/** unsigned int reduce_TBAssignment_tbdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_TBAssignment_tbdefault_id_variable();



/** unsigned int count_TBAssignment_tbdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_TBAssignment_tbdefault_id_variable(int count_value);

/** unsigned int min_TBAssignment_tbdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_TBAssignment_tbdefault_id_variable();
/** unsigned int max_TBAssignment_tbdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_TBAssignment_tbdefault_id_variable();

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

/** float reduce_Household_hhdefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Household_hhdefault_lambda_variable();



/** float min_Household_hhdefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Household_hhdefault_lambda_variable();
/** float max_Household_hhdefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Household_hhdefault_lambda_variable();

/** unsigned int reduce_Household_hhdefault_active_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Household_hhdefault_active_variable();



/** unsigned int count_Household_hhdefault_active_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Household_hhdefault_active_variable(int count_value);

/** unsigned int min_Household_hhdefault_active_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Household_hhdefault_active_variable();
/** unsigned int max_Household_hhdefault_active_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Household_hhdefault_active_variable();

/** unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_id_variable();



/** unsigned int count_HouseholdMembership_hhmembershipdefault_household_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_HouseholdMembership_hhmembershipdefault_household_id_variable(int count_value);

/** unsigned int min_HouseholdMembership_hhmembershipdefault_household_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_HouseholdMembership_hhmembershipdefault_household_id_variable();
/** unsigned int max_HouseholdMembership_hhmembershipdefault_household_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_HouseholdMembership_hhmembershipdefault_household_id_variable();

/** unsigned int reduce_HouseholdMembership_hhmembershipdefault_person_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_HouseholdMembership_hhmembershipdefault_person_id_variable();



/** unsigned int count_HouseholdMembership_hhmembershipdefault_person_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_HouseholdMembership_hhmembershipdefault_person_id_variable(int count_value);

/** unsigned int min_HouseholdMembership_hhmembershipdefault_person_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_HouseholdMembership_hhmembershipdefault_person_id_variable();
/** unsigned int max_HouseholdMembership_hhmembershipdefault_person_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_HouseholdMembership_hhmembershipdefault_person_id_variable();

/** unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_size_variable();



/** unsigned int count_HouseholdMembership_hhmembershipdefault_household_size_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_HouseholdMembership_hhmembershipdefault_household_size_variable(int count_value);

/** unsigned int min_HouseholdMembership_hhmembershipdefault_household_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_HouseholdMembership_hhmembershipdefault_household_size_variable();
/** unsigned int max_HouseholdMembership_hhmembershipdefault_household_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_HouseholdMembership_hhmembershipdefault_household_size_variable();

/** unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchgoing_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchgoing_variable();



/** unsigned int count_HouseholdMembership_hhmembershipdefault_churchgoing_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_HouseholdMembership_hhmembershipdefault_churchgoing_variable(int count_value);

/** unsigned int min_HouseholdMembership_hhmembershipdefault_churchgoing_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_HouseholdMembership_hhmembershipdefault_churchgoing_variable();
/** unsigned int max_HouseholdMembership_hhmembershipdefault_churchgoing_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_HouseholdMembership_hhmembershipdefault_churchgoing_variable();

/** unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchfreq_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_HouseholdMembership_hhmembershipdefault_churchfreq_variable();



/** unsigned int count_HouseholdMembership_hhmembershipdefault_churchfreq_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_HouseholdMembership_hhmembershipdefault_churchfreq_variable(int count_value);

/** unsigned int min_HouseholdMembership_hhmembershipdefault_churchfreq_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_HouseholdMembership_hhmembershipdefault_churchfreq_variable();
/** unsigned int max_HouseholdMembership_hhmembershipdefault_churchfreq_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_HouseholdMembership_hhmembershipdefault_churchfreq_variable();

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

/** float reduce_Church_chudefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Church_chudefault_lambda_variable();



/** float min_Church_chudefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Church_chudefault_lambda_variable();
/** float max_Church_chudefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Church_chudefault_lambda_variable();

/** unsigned int reduce_Church_chudefault_active_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Church_chudefault_active_variable();



/** unsigned int count_Church_chudefault_active_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Church_chudefault_active_variable(int count_value);

/** unsigned int min_Church_chudefault_active_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Church_chudefault_active_variable();
/** unsigned int max_Church_chudefault_active_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Church_chudefault_active_variable();

/** unsigned int reduce_ChurchMembership_chumembershipdefault_church_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_ChurchMembership_chumembershipdefault_church_id_variable();



/** unsigned int count_ChurchMembership_chumembershipdefault_church_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_ChurchMembership_chumembershipdefault_church_id_variable(int count_value);

/** unsigned int min_ChurchMembership_chumembershipdefault_church_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_ChurchMembership_chumembershipdefault_church_id_variable();
/** unsigned int max_ChurchMembership_chumembershipdefault_church_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_ChurchMembership_chumembershipdefault_church_id_variable();

/** unsigned int reduce_ChurchMembership_chumembershipdefault_household_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_ChurchMembership_chumembershipdefault_household_id_variable();



/** unsigned int count_ChurchMembership_chumembershipdefault_household_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_ChurchMembership_chumembershipdefault_household_id_variable(int count_value);

/** unsigned int min_ChurchMembership_chumembershipdefault_household_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_ChurchMembership_chumembershipdefault_household_id_variable();
/** unsigned int max_ChurchMembership_chumembershipdefault_household_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_ChurchMembership_chumembershipdefault_household_id_variable();

/** float reduce_ChurchMembership_chumembershipdefault_churchdur_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_ChurchMembership_chumembershipdefault_churchdur_variable();



/** float min_ChurchMembership_chumembershipdefault_churchdur_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_ChurchMembership_chumembershipdefault_churchdur_variable();
/** float max_ChurchMembership_chumembershipdefault_churchdur_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_ChurchMembership_chumembershipdefault_churchdur_variable();

/** unsigned int reduce_Transport_trdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Transport_trdefault_id_variable();



/** unsigned int count_Transport_trdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Transport_trdefault_id_variable(int count_value);

/** unsigned int min_Transport_trdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Transport_trdefault_id_variable();
/** unsigned int max_Transport_trdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Transport_trdefault_id_variable();

/** float reduce_Transport_trdefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Transport_trdefault_lambda_variable();



/** float min_Transport_trdefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Transport_trdefault_lambda_variable();
/** float max_Transport_trdefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Transport_trdefault_lambda_variable();

/** unsigned int reduce_Transport_trdefault_active_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Transport_trdefault_active_variable();



/** unsigned int count_Transport_trdefault_active_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Transport_trdefault_active_variable(int count_value);

/** unsigned int min_Transport_trdefault_active_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Transport_trdefault_active_variable();
/** unsigned int max_Transport_trdefault_active_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Transport_trdefault_active_variable();

/** int reduce_TransportMembership_trmembershipdefault_person_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_TransportMembership_trmembershipdefault_person_id_variable();



/** int count_TransportMembership_trmembershipdefault_person_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_TransportMembership_trmembershipdefault_person_id_variable(int count_value);

/** int min_TransportMembership_trmembershipdefault_person_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_TransportMembership_trmembershipdefault_person_id_variable();
/** int max_TransportMembership_trmembershipdefault_person_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_TransportMembership_trmembershipdefault_person_id_variable();

/** unsigned int reduce_TransportMembership_trmembershipdefault_transport_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_TransportMembership_trmembershipdefault_transport_id_variable();



/** unsigned int count_TransportMembership_trmembershipdefault_transport_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_TransportMembership_trmembershipdefault_transport_id_variable(int count_value);

/** unsigned int min_TransportMembership_trmembershipdefault_transport_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_TransportMembership_trmembershipdefault_transport_id_variable();
/** unsigned int max_TransportMembership_trmembershipdefault_transport_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_TransportMembership_trmembershipdefault_transport_id_variable();

/** unsigned int reduce_TransportMembership_trmembershipdefault_duration_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_TransportMembership_trmembershipdefault_duration_variable();



/** unsigned int count_TransportMembership_trmembershipdefault_duration_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_TransportMembership_trmembershipdefault_duration_variable(int count_value);

/** unsigned int min_TransportMembership_trmembershipdefault_duration_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_TransportMembership_trmembershipdefault_duration_variable();
/** unsigned int max_TransportMembership_trmembershipdefault_duration_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_TransportMembership_trmembershipdefault_duration_variable();

/** unsigned int reduce_Clinic_cldefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Clinic_cldefault_id_variable();



/** unsigned int count_Clinic_cldefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Clinic_cldefault_id_variable(int count_value);

/** unsigned int min_Clinic_cldefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Clinic_cldefault_id_variable();
/** unsigned int max_Clinic_cldefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Clinic_cldefault_id_variable();

/** float reduce_Clinic_cldefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Clinic_cldefault_lambda_variable();



/** float min_Clinic_cldefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Clinic_cldefault_lambda_variable();
/** float max_Clinic_cldefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Clinic_cldefault_lambda_variable();

/** unsigned int reduce_Workplace_wpdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Workplace_wpdefault_id_variable();



/** unsigned int count_Workplace_wpdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Workplace_wpdefault_id_variable(int count_value);

/** unsigned int min_Workplace_wpdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Workplace_wpdefault_id_variable();
/** unsigned int max_Workplace_wpdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Workplace_wpdefault_id_variable();

/** float reduce_Workplace_wpdefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Workplace_wpdefault_lambda_variable();



/** float min_Workplace_wpdefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Workplace_wpdefault_lambda_variable();
/** float max_Workplace_wpdefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Workplace_wpdefault_lambda_variable();

/** unsigned int reduce_WorkplaceMembership_wpmembershipdefault_person_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_WorkplaceMembership_wpmembershipdefault_person_id_variable();



/** unsigned int count_WorkplaceMembership_wpmembershipdefault_person_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_WorkplaceMembership_wpmembershipdefault_person_id_variable(int count_value);

/** unsigned int min_WorkplaceMembership_wpmembershipdefault_person_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_WorkplaceMembership_wpmembershipdefault_person_id_variable();
/** unsigned int max_WorkplaceMembership_wpmembershipdefault_person_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_WorkplaceMembership_wpmembershipdefault_person_id_variable();

/** unsigned int reduce_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();



/** unsigned int count_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(int count_value);

/** unsigned int min_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();
/** unsigned int max_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_WorkplaceMembership_wpmembershipdefault_workplace_id_variable();

/** unsigned int reduce_Bar_bdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Bar_bdefault_id_variable();



/** unsigned int count_Bar_bdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Bar_bdefault_id_variable(int count_value);

/** unsigned int min_Bar_bdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Bar_bdefault_id_variable();
/** unsigned int max_Bar_bdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Bar_bdefault_id_variable();

/** float reduce_Bar_bdefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Bar_bdefault_lambda_variable();



/** float min_Bar_bdefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Bar_bdefault_lambda_variable();
/** float max_Bar_bdefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Bar_bdefault_lambda_variable();

/** unsigned int reduce_School_schdefault_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_School_schdefault_id_variable();



/** unsigned int count_School_schdefault_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_School_schdefault_id_variable(int count_value);

/** unsigned int min_School_schdefault_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_School_schdefault_id_variable();
/** unsigned int max_School_schdefault_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_School_schdefault_id_variable();

/** float reduce_School_schdefault_lambda_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_School_schdefault_lambda_variable();



/** float min_School_schdefault_lambda_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_School_schdefault_lambda_variable();
/** float max_School_schdefault_lambda_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_School_schdefault_lambda_variable();

/** unsigned int reduce_SchoolMembership_schmembershipdefault_person_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_SchoolMembership_schmembershipdefault_person_id_variable();



/** unsigned int count_SchoolMembership_schmembershipdefault_person_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_SchoolMembership_schmembershipdefault_person_id_variable(int count_value);

/** unsigned int min_SchoolMembership_schmembershipdefault_person_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_SchoolMembership_schmembershipdefault_person_id_variable();
/** unsigned int max_SchoolMembership_schmembershipdefault_person_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_SchoolMembership_schmembershipdefault_person_id_variable();

/** unsigned int reduce_SchoolMembership_schmembershipdefault_school_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_SchoolMembership_schmembershipdefault_school_id_variable();



/** unsigned int count_SchoolMembership_schmembershipdefault_school_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_SchoolMembership_schmembershipdefault_school_id_variable(int count_value);

/** unsigned int min_SchoolMembership_schmembershipdefault_school_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_SchoolMembership_schmembershipdefault_school_id_variable();
/** unsigned int max_SchoolMembership_schmembershipdefault_school_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_SchoolMembership_schmembershipdefault_school_id_variable();


  
/* global constant variables */

__constant__ float TIME_STEP;

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

__constant__ float CHURCH_PROPORTION;

__constant__ float TRANSPORT_BETA0;

__constant__ float TRANSPORT_BETA1;

__constant__ float TRANSPORT_FREQ0;

__constant__ float TRANSPORT_FREQ2;

__constant__ float TRANSPORT_DUR20;

__constant__ float TRANSPORT_DUR45;

__constant__ unsigned int TRANSPORT_SIZE;

__constant__ float HIV_PREVALENCE;

__constant__ float ART_COVERAGE;

__constant__ float RR_HIV;

__constant__ float RR_ART;

__constant__ float TB_PREVALENCE;

__constant__ float DEFAULT_M_P;

__constant__ float DEFAULT_F_P;

__constant__ float DEFAULT_Q;

__constant__ unsigned int DEFAULT_K;

__constant__ float THETA;

__constant__ float TRANSPORT_A;

__constant__ float CHURCH_A;

__constant__ float CLINIC_A;

__constant__ float HOUSEHOLD_A;

__constant__ float TRANSPORT_V;

__constant__ float HOUSEHOLD_V;

__constant__ float CLINIC_V;

__constant__ float CHURCH_V_MULTIPLIER;

__constant__ float WORKPLACE_BETA0;

__constant__ float WORKPLACE_BETAA;

__constant__ float WORKPLACE_BETAS;

__constant__ float WORKPLACE_BETAAS;

__constant__ float WORKPLACE_A;

__constant__ unsigned int WORKPLACE_DUR;

__constant__ unsigned int WORKPLACE_SIZE;

__constant__ float WORKPLACE_V;

__constant__ unsigned int HOUSEHOLDS;

__constant__ unsigned int BARS;

__constant__ float RR_AS_F_46;

__constant__ float RR_AS_F_26;

__constant__ float RR_AS_F_18;

__constant__ float RR_AS_M_46;

__constant__ float RR_AS_M_26;

__constant__ float RR_AS_M_18;

__constant__ float BAR_BETA0;

__constant__ float BAR_BETAA;

__constant__ float BAR_BETAS;

__constant__ float BAR_BETAAS;

__constant__ unsigned int BAR_SIZE;

__constant__ unsigned int SCHOOL_SIZE;

__constant__ float BAR_A;

__constant__ float BAR_V;

__constant__ float SCHOOL_A;

__constant__ float SCHOOL_V;

__constant__ unsigned int SEED;

__constant__ float HOUSEHOLD_EXP;

__constant__ float CHURCH_EXP;

__constant__ float TRANSPORT_EXP;

__constant__ float CLINIC_EXP;

__constant__ float WORKPLACE_EXP;

__constant__ float BAR_EXP;

__constant__ float SCHOOL_EXP;

__constant__ float PROB;

__constant__ float BAR_M_PROB1;

__constant__ float BAR_M_PROB2;

__constant__ float BAR_M_PROB3;

__constant__ float BAR_M_PROB4;

__constant__ float BAR_M_PROB5;

__constant__ float BAR_M_PROB7;

__constant__ float BAR_F_PROB1;

__constant__ float BAR_F_PROB2;

__constant__ float BAR_F_PROB3;

__constant__ float BAR_F_PROB4;

__constant__ float BAR_F_PROB5;

__constant__ float BAR_F_PROB7;

__constant__ float CLINIC_DUR;

__constant__ float BAR_DUR;

__constant__ float SCHOOL_DUR;

__constant__ float VISITING_DUR;

__constant__ unsigned int OUTPUT_ID;

/** set_TIME_STEP
 * Sets the constant variable TIME_STEP on the device which can then be used in the agent functions.
 * @param h_TIME_STEP value to set the variable
 */
extern void set_TIME_STEP(float* h_TIME_STEP);

extern const float* get_TIME_STEP();


extern float h_env_TIME_STEP;

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

/** set_CHURCH_PROPORTION
 * Sets the constant variable CHURCH_PROPORTION on the device which can then be used in the agent functions.
 * @param h_CHURCH_PROPORTION value to set the variable
 */
extern void set_CHURCH_PROPORTION(float* h_CHURCH_PROPORTION);

extern const float* get_CHURCH_PROPORTION();


extern float h_env_CHURCH_PROPORTION;

/** set_TRANSPORT_BETA0
 * Sets the constant variable TRANSPORT_BETA0 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_BETA0 value to set the variable
 */
extern void set_TRANSPORT_BETA0(float* h_TRANSPORT_BETA0);

extern const float* get_TRANSPORT_BETA0();


extern float h_env_TRANSPORT_BETA0;

/** set_TRANSPORT_BETA1
 * Sets the constant variable TRANSPORT_BETA1 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_BETA1 value to set the variable
 */
extern void set_TRANSPORT_BETA1(float* h_TRANSPORT_BETA1);

extern const float* get_TRANSPORT_BETA1();


extern float h_env_TRANSPORT_BETA1;

/** set_TRANSPORT_FREQ0
 * Sets the constant variable TRANSPORT_FREQ0 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_FREQ0 value to set the variable
 */
extern void set_TRANSPORT_FREQ0(float* h_TRANSPORT_FREQ0);

extern const float* get_TRANSPORT_FREQ0();


extern float h_env_TRANSPORT_FREQ0;

/** set_TRANSPORT_FREQ2
 * Sets the constant variable TRANSPORT_FREQ2 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_FREQ2 value to set the variable
 */
extern void set_TRANSPORT_FREQ2(float* h_TRANSPORT_FREQ2);

extern const float* get_TRANSPORT_FREQ2();


extern float h_env_TRANSPORT_FREQ2;

/** set_TRANSPORT_DUR20
 * Sets the constant variable TRANSPORT_DUR20 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_DUR20 value to set the variable
 */
extern void set_TRANSPORT_DUR20(float* h_TRANSPORT_DUR20);

extern const float* get_TRANSPORT_DUR20();


extern float h_env_TRANSPORT_DUR20;

/** set_TRANSPORT_DUR45
 * Sets the constant variable TRANSPORT_DUR45 on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_DUR45 value to set the variable
 */
extern void set_TRANSPORT_DUR45(float* h_TRANSPORT_DUR45);

extern const float* get_TRANSPORT_DUR45();


extern float h_env_TRANSPORT_DUR45;

/** set_TRANSPORT_SIZE
 * Sets the constant variable TRANSPORT_SIZE on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_SIZE value to set the variable
 */
extern void set_TRANSPORT_SIZE(unsigned int* h_TRANSPORT_SIZE);

extern const unsigned int* get_TRANSPORT_SIZE();


extern unsigned int h_env_TRANSPORT_SIZE;

/** set_HIV_PREVALENCE
 * Sets the constant variable HIV_PREVALENCE on the device which can then be used in the agent functions.
 * @param h_HIV_PREVALENCE value to set the variable
 */
extern void set_HIV_PREVALENCE(float* h_HIV_PREVALENCE);

extern const float* get_HIV_PREVALENCE();


extern float h_env_HIV_PREVALENCE;

/** set_ART_COVERAGE
 * Sets the constant variable ART_COVERAGE on the device which can then be used in the agent functions.
 * @param h_ART_COVERAGE value to set the variable
 */
extern void set_ART_COVERAGE(float* h_ART_COVERAGE);

extern const float* get_ART_COVERAGE();


extern float h_env_ART_COVERAGE;

/** set_RR_HIV
 * Sets the constant variable RR_HIV on the device which can then be used in the agent functions.
 * @param h_RR_HIV value to set the variable
 */
extern void set_RR_HIV(float* h_RR_HIV);

extern const float* get_RR_HIV();


extern float h_env_RR_HIV;

/** set_RR_ART
 * Sets the constant variable RR_ART on the device which can then be used in the agent functions.
 * @param h_RR_ART value to set the variable
 */
extern void set_RR_ART(float* h_RR_ART);

extern const float* get_RR_ART();


extern float h_env_RR_ART;

/** set_TB_PREVALENCE
 * Sets the constant variable TB_PREVALENCE on the device which can then be used in the agent functions.
 * @param h_TB_PREVALENCE value to set the variable
 */
extern void set_TB_PREVALENCE(float* h_TB_PREVALENCE);

extern const float* get_TB_PREVALENCE();


extern float h_env_TB_PREVALENCE;

/** set_DEFAULT_M_P
 * Sets the constant variable DEFAULT_M_P on the device which can then be used in the agent functions.
 * @param h_DEFAULT_M_P value to set the variable
 */
extern void set_DEFAULT_M_P(float* h_DEFAULT_M_P);

extern const float* get_DEFAULT_M_P();


extern float h_env_DEFAULT_M_P;

/** set_DEFAULT_F_P
 * Sets the constant variable DEFAULT_F_P on the device which can then be used in the agent functions.
 * @param h_DEFAULT_F_P value to set the variable
 */
extern void set_DEFAULT_F_P(float* h_DEFAULT_F_P);

extern const float* get_DEFAULT_F_P();


extern float h_env_DEFAULT_F_P;

/** set_DEFAULT_Q
 * Sets the constant variable DEFAULT_Q on the device which can then be used in the agent functions.
 * @param h_DEFAULT_Q value to set the variable
 */
extern void set_DEFAULT_Q(float* h_DEFAULT_Q);

extern const float* get_DEFAULT_Q();


extern float h_env_DEFAULT_Q;

/** set_DEFAULT_K
 * Sets the constant variable DEFAULT_K on the device which can then be used in the agent functions.
 * @param h_DEFAULT_K value to set the variable
 */
extern void set_DEFAULT_K(unsigned int* h_DEFAULT_K);

extern const unsigned int* get_DEFAULT_K();


extern unsigned int h_env_DEFAULT_K;

/** set_THETA
 * Sets the constant variable THETA on the device which can then be used in the agent functions.
 * @param h_THETA value to set the variable
 */
extern void set_THETA(float* h_THETA);

extern const float* get_THETA();


extern float h_env_THETA;

/** set_TRANSPORT_A
 * Sets the constant variable TRANSPORT_A on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_A value to set the variable
 */
extern void set_TRANSPORT_A(float* h_TRANSPORT_A);

extern const float* get_TRANSPORT_A();


extern float h_env_TRANSPORT_A;

/** set_CHURCH_A
 * Sets the constant variable CHURCH_A on the device which can then be used in the agent functions.
 * @param h_CHURCH_A value to set the variable
 */
extern void set_CHURCH_A(float* h_CHURCH_A);

extern const float* get_CHURCH_A();


extern float h_env_CHURCH_A;

/** set_CLINIC_A
 * Sets the constant variable CLINIC_A on the device which can then be used in the agent functions.
 * @param h_CLINIC_A value to set the variable
 */
extern void set_CLINIC_A(float* h_CLINIC_A);

extern const float* get_CLINIC_A();


extern float h_env_CLINIC_A;

/** set_HOUSEHOLD_A
 * Sets the constant variable HOUSEHOLD_A on the device which can then be used in the agent functions.
 * @param h_HOUSEHOLD_A value to set the variable
 */
extern void set_HOUSEHOLD_A(float* h_HOUSEHOLD_A);

extern const float* get_HOUSEHOLD_A();


extern float h_env_HOUSEHOLD_A;

/** set_TRANSPORT_V
 * Sets the constant variable TRANSPORT_V on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_V value to set the variable
 */
extern void set_TRANSPORT_V(float* h_TRANSPORT_V);

extern const float* get_TRANSPORT_V();


extern float h_env_TRANSPORT_V;

/** set_HOUSEHOLD_V
 * Sets the constant variable HOUSEHOLD_V on the device which can then be used in the agent functions.
 * @param h_HOUSEHOLD_V value to set the variable
 */
extern void set_HOUSEHOLD_V(float* h_HOUSEHOLD_V);

extern const float* get_HOUSEHOLD_V();


extern float h_env_HOUSEHOLD_V;

/** set_CLINIC_V
 * Sets the constant variable CLINIC_V on the device which can then be used in the agent functions.
 * @param h_CLINIC_V value to set the variable
 */
extern void set_CLINIC_V(float* h_CLINIC_V);

extern const float* get_CLINIC_V();


extern float h_env_CLINIC_V;

/** set_CHURCH_V_MULTIPLIER
 * Sets the constant variable CHURCH_V_MULTIPLIER on the device which can then be used in the agent functions.
 * @param h_CHURCH_V_MULTIPLIER value to set the variable
 */
extern void set_CHURCH_V_MULTIPLIER(float* h_CHURCH_V_MULTIPLIER);

extern const float* get_CHURCH_V_MULTIPLIER();


extern float h_env_CHURCH_V_MULTIPLIER;

/** set_WORKPLACE_BETA0
 * Sets the constant variable WORKPLACE_BETA0 on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_BETA0 value to set the variable
 */
extern void set_WORKPLACE_BETA0(float* h_WORKPLACE_BETA0);

extern const float* get_WORKPLACE_BETA0();


extern float h_env_WORKPLACE_BETA0;

/** set_WORKPLACE_BETAA
 * Sets the constant variable WORKPLACE_BETAA on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_BETAA value to set the variable
 */
extern void set_WORKPLACE_BETAA(float* h_WORKPLACE_BETAA);

extern const float* get_WORKPLACE_BETAA();


extern float h_env_WORKPLACE_BETAA;

/** set_WORKPLACE_BETAS
 * Sets the constant variable WORKPLACE_BETAS on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_BETAS value to set the variable
 */
extern void set_WORKPLACE_BETAS(float* h_WORKPLACE_BETAS);

extern const float* get_WORKPLACE_BETAS();


extern float h_env_WORKPLACE_BETAS;

/** set_WORKPLACE_BETAAS
 * Sets the constant variable WORKPLACE_BETAAS on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_BETAAS value to set the variable
 */
extern void set_WORKPLACE_BETAAS(float* h_WORKPLACE_BETAAS);

extern const float* get_WORKPLACE_BETAAS();


extern float h_env_WORKPLACE_BETAAS;

/** set_WORKPLACE_A
 * Sets the constant variable WORKPLACE_A on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_A value to set the variable
 */
extern void set_WORKPLACE_A(float* h_WORKPLACE_A);

extern const float* get_WORKPLACE_A();


extern float h_env_WORKPLACE_A;

/** set_WORKPLACE_DUR
 * Sets the constant variable WORKPLACE_DUR on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_DUR value to set the variable
 */
extern void set_WORKPLACE_DUR(unsigned int* h_WORKPLACE_DUR);

extern const unsigned int* get_WORKPLACE_DUR();


extern unsigned int h_env_WORKPLACE_DUR;

/** set_WORKPLACE_SIZE
 * Sets the constant variable WORKPLACE_SIZE on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_SIZE value to set the variable
 */
extern void set_WORKPLACE_SIZE(unsigned int* h_WORKPLACE_SIZE);

extern const unsigned int* get_WORKPLACE_SIZE();


extern unsigned int h_env_WORKPLACE_SIZE;

/** set_WORKPLACE_V
 * Sets the constant variable WORKPLACE_V on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_V value to set the variable
 */
extern void set_WORKPLACE_V(float* h_WORKPLACE_V);

extern const float* get_WORKPLACE_V();


extern float h_env_WORKPLACE_V;

/** set_HOUSEHOLDS
 * Sets the constant variable HOUSEHOLDS on the device which can then be used in the agent functions.
 * @param h_HOUSEHOLDS value to set the variable
 */
extern void set_HOUSEHOLDS(unsigned int* h_HOUSEHOLDS);

extern const unsigned int* get_HOUSEHOLDS();


extern unsigned int h_env_HOUSEHOLDS;

/** set_BARS
 * Sets the constant variable BARS on the device which can then be used in the agent functions.
 * @param h_BARS value to set the variable
 */
extern void set_BARS(unsigned int* h_BARS);

extern const unsigned int* get_BARS();


extern unsigned int h_env_BARS;

/** set_RR_AS_F_46
 * Sets the constant variable RR_AS_F_46 on the device which can then be used in the agent functions.
 * @param h_RR_AS_F_46 value to set the variable
 */
extern void set_RR_AS_F_46(float* h_RR_AS_F_46);

extern const float* get_RR_AS_F_46();


extern float h_env_RR_AS_F_46;

/** set_RR_AS_F_26
 * Sets the constant variable RR_AS_F_26 on the device which can then be used in the agent functions.
 * @param h_RR_AS_F_26 value to set the variable
 */
extern void set_RR_AS_F_26(float* h_RR_AS_F_26);

extern const float* get_RR_AS_F_26();


extern float h_env_RR_AS_F_26;

/** set_RR_AS_F_18
 * Sets the constant variable RR_AS_F_18 on the device which can then be used in the agent functions.
 * @param h_RR_AS_F_18 value to set the variable
 */
extern void set_RR_AS_F_18(float* h_RR_AS_F_18);

extern const float* get_RR_AS_F_18();


extern float h_env_RR_AS_F_18;

/** set_RR_AS_M_46
 * Sets the constant variable RR_AS_M_46 on the device which can then be used in the agent functions.
 * @param h_RR_AS_M_46 value to set the variable
 */
extern void set_RR_AS_M_46(float* h_RR_AS_M_46);

extern const float* get_RR_AS_M_46();


extern float h_env_RR_AS_M_46;

/** set_RR_AS_M_26
 * Sets the constant variable RR_AS_M_26 on the device which can then be used in the agent functions.
 * @param h_RR_AS_M_26 value to set the variable
 */
extern void set_RR_AS_M_26(float* h_RR_AS_M_26);

extern const float* get_RR_AS_M_26();


extern float h_env_RR_AS_M_26;

/** set_RR_AS_M_18
 * Sets the constant variable RR_AS_M_18 on the device which can then be used in the agent functions.
 * @param h_RR_AS_M_18 value to set the variable
 */
extern void set_RR_AS_M_18(float* h_RR_AS_M_18);

extern const float* get_RR_AS_M_18();


extern float h_env_RR_AS_M_18;

/** set_BAR_BETA0
 * Sets the constant variable BAR_BETA0 on the device which can then be used in the agent functions.
 * @param h_BAR_BETA0 value to set the variable
 */
extern void set_BAR_BETA0(float* h_BAR_BETA0);

extern const float* get_BAR_BETA0();


extern float h_env_BAR_BETA0;

/** set_BAR_BETAA
 * Sets the constant variable BAR_BETAA on the device which can then be used in the agent functions.
 * @param h_BAR_BETAA value to set the variable
 */
extern void set_BAR_BETAA(float* h_BAR_BETAA);

extern const float* get_BAR_BETAA();


extern float h_env_BAR_BETAA;

/** set_BAR_BETAS
 * Sets the constant variable BAR_BETAS on the device which can then be used in the agent functions.
 * @param h_BAR_BETAS value to set the variable
 */
extern void set_BAR_BETAS(float* h_BAR_BETAS);

extern const float* get_BAR_BETAS();


extern float h_env_BAR_BETAS;

/** set_BAR_BETAAS
 * Sets the constant variable BAR_BETAAS on the device which can then be used in the agent functions.
 * @param h_BAR_BETAAS value to set the variable
 */
extern void set_BAR_BETAAS(float* h_BAR_BETAAS);

extern const float* get_BAR_BETAAS();


extern float h_env_BAR_BETAAS;

/** set_BAR_SIZE
 * Sets the constant variable BAR_SIZE on the device which can then be used in the agent functions.
 * @param h_BAR_SIZE value to set the variable
 */
extern void set_BAR_SIZE(unsigned int* h_BAR_SIZE);

extern const unsigned int* get_BAR_SIZE();


extern unsigned int h_env_BAR_SIZE;

/** set_SCHOOL_SIZE
 * Sets the constant variable SCHOOL_SIZE on the device which can then be used in the agent functions.
 * @param h_SCHOOL_SIZE value to set the variable
 */
extern void set_SCHOOL_SIZE(unsigned int* h_SCHOOL_SIZE);

extern const unsigned int* get_SCHOOL_SIZE();


extern unsigned int h_env_SCHOOL_SIZE;

/** set_BAR_A
 * Sets the constant variable BAR_A on the device which can then be used in the agent functions.
 * @param h_BAR_A value to set the variable
 */
extern void set_BAR_A(float* h_BAR_A);

extern const float* get_BAR_A();


extern float h_env_BAR_A;

/** set_BAR_V
 * Sets the constant variable BAR_V on the device which can then be used in the agent functions.
 * @param h_BAR_V value to set the variable
 */
extern void set_BAR_V(float* h_BAR_V);

extern const float* get_BAR_V();


extern float h_env_BAR_V;

/** set_SCHOOL_A
 * Sets the constant variable SCHOOL_A on the device which can then be used in the agent functions.
 * @param h_SCHOOL_A value to set the variable
 */
extern void set_SCHOOL_A(float* h_SCHOOL_A);

extern const float* get_SCHOOL_A();


extern float h_env_SCHOOL_A;

/** set_SCHOOL_V
 * Sets the constant variable SCHOOL_V on the device which can then be used in the agent functions.
 * @param h_SCHOOL_V value to set the variable
 */
extern void set_SCHOOL_V(float* h_SCHOOL_V);

extern const float* get_SCHOOL_V();


extern float h_env_SCHOOL_V;

/** set_SEED
 * Sets the constant variable SEED on the device which can then be used in the agent functions.
 * @param h_SEED value to set the variable
 */
extern void set_SEED(unsigned int* h_SEED);

extern const unsigned int* get_SEED();


extern unsigned int h_env_SEED;

/** set_HOUSEHOLD_EXP
 * Sets the constant variable HOUSEHOLD_EXP on the device which can then be used in the agent functions.
 * @param h_HOUSEHOLD_EXP value to set the variable
 */
extern void set_HOUSEHOLD_EXP(float* h_HOUSEHOLD_EXP);

extern const float* get_HOUSEHOLD_EXP();


extern float h_env_HOUSEHOLD_EXP;

/** set_CHURCH_EXP
 * Sets the constant variable CHURCH_EXP on the device which can then be used in the agent functions.
 * @param h_CHURCH_EXP value to set the variable
 */
extern void set_CHURCH_EXP(float* h_CHURCH_EXP);

extern const float* get_CHURCH_EXP();


extern float h_env_CHURCH_EXP;

/** set_TRANSPORT_EXP
 * Sets the constant variable TRANSPORT_EXP on the device which can then be used in the agent functions.
 * @param h_TRANSPORT_EXP value to set the variable
 */
extern void set_TRANSPORT_EXP(float* h_TRANSPORT_EXP);

extern const float* get_TRANSPORT_EXP();


extern float h_env_TRANSPORT_EXP;

/** set_CLINIC_EXP
 * Sets the constant variable CLINIC_EXP on the device which can then be used in the agent functions.
 * @param h_CLINIC_EXP value to set the variable
 */
extern void set_CLINIC_EXP(float* h_CLINIC_EXP);

extern const float* get_CLINIC_EXP();


extern float h_env_CLINIC_EXP;

/** set_WORKPLACE_EXP
 * Sets the constant variable WORKPLACE_EXP on the device which can then be used in the agent functions.
 * @param h_WORKPLACE_EXP value to set the variable
 */
extern void set_WORKPLACE_EXP(float* h_WORKPLACE_EXP);

extern const float* get_WORKPLACE_EXP();


extern float h_env_WORKPLACE_EXP;

/** set_BAR_EXP
 * Sets the constant variable BAR_EXP on the device which can then be used in the agent functions.
 * @param h_BAR_EXP value to set the variable
 */
extern void set_BAR_EXP(float* h_BAR_EXP);

extern const float* get_BAR_EXP();


extern float h_env_BAR_EXP;

/** set_SCHOOL_EXP
 * Sets the constant variable SCHOOL_EXP on the device which can then be used in the agent functions.
 * @param h_SCHOOL_EXP value to set the variable
 */
extern void set_SCHOOL_EXP(float* h_SCHOOL_EXP);

extern const float* get_SCHOOL_EXP();


extern float h_env_SCHOOL_EXP;

/** set_PROB
 * Sets the constant variable PROB on the device which can then be used in the agent functions.
 * @param h_PROB value to set the variable
 */
extern void set_PROB(float* h_PROB);

extern const float* get_PROB();


extern float h_env_PROB;

/** set_BAR_M_PROB1
 * Sets the constant variable BAR_M_PROB1 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB1 value to set the variable
 */
extern void set_BAR_M_PROB1(float* h_BAR_M_PROB1);

extern const float* get_BAR_M_PROB1();


extern float h_env_BAR_M_PROB1;

/** set_BAR_M_PROB2
 * Sets the constant variable BAR_M_PROB2 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB2 value to set the variable
 */
extern void set_BAR_M_PROB2(float* h_BAR_M_PROB2);

extern const float* get_BAR_M_PROB2();


extern float h_env_BAR_M_PROB2;

/** set_BAR_M_PROB3
 * Sets the constant variable BAR_M_PROB3 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB3 value to set the variable
 */
extern void set_BAR_M_PROB3(float* h_BAR_M_PROB3);

extern const float* get_BAR_M_PROB3();


extern float h_env_BAR_M_PROB3;

/** set_BAR_M_PROB4
 * Sets the constant variable BAR_M_PROB4 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB4 value to set the variable
 */
extern void set_BAR_M_PROB4(float* h_BAR_M_PROB4);

extern const float* get_BAR_M_PROB4();


extern float h_env_BAR_M_PROB4;

/** set_BAR_M_PROB5
 * Sets the constant variable BAR_M_PROB5 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB5 value to set the variable
 */
extern void set_BAR_M_PROB5(float* h_BAR_M_PROB5);

extern const float* get_BAR_M_PROB5();


extern float h_env_BAR_M_PROB5;

/** set_BAR_M_PROB7
 * Sets the constant variable BAR_M_PROB7 on the device which can then be used in the agent functions.
 * @param h_BAR_M_PROB7 value to set the variable
 */
extern void set_BAR_M_PROB7(float* h_BAR_M_PROB7);

extern const float* get_BAR_M_PROB7();


extern float h_env_BAR_M_PROB7;

/** set_BAR_F_PROB1
 * Sets the constant variable BAR_F_PROB1 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB1 value to set the variable
 */
extern void set_BAR_F_PROB1(float* h_BAR_F_PROB1);

extern const float* get_BAR_F_PROB1();


extern float h_env_BAR_F_PROB1;

/** set_BAR_F_PROB2
 * Sets the constant variable BAR_F_PROB2 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB2 value to set the variable
 */
extern void set_BAR_F_PROB2(float* h_BAR_F_PROB2);

extern const float* get_BAR_F_PROB2();


extern float h_env_BAR_F_PROB2;

/** set_BAR_F_PROB3
 * Sets the constant variable BAR_F_PROB3 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB3 value to set the variable
 */
extern void set_BAR_F_PROB3(float* h_BAR_F_PROB3);

extern const float* get_BAR_F_PROB3();


extern float h_env_BAR_F_PROB3;

/** set_BAR_F_PROB4
 * Sets the constant variable BAR_F_PROB4 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB4 value to set the variable
 */
extern void set_BAR_F_PROB4(float* h_BAR_F_PROB4);

extern const float* get_BAR_F_PROB4();


extern float h_env_BAR_F_PROB4;

/** set_BAR_F_PROB5
 * Sets the constant variable BAR_F_PROB5 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB5 value to set the variable
 */
extern void set_BAR_F_PROB5(float* h_BAR_F_PROB5);

extern const float* get_BAR_F_PROB5();


extern float h_env_BAR_F_PROB5;

/** set_BAR_F_PROB7
 * Sets the constant variable BAR_F_PROB7 on the device which can then be used in the agent functions.
 * @param h_BAR_F_PROB7 value to set the variable
 */
extern void set_BAR_F_PROB7(float* h_BAR_F_PROB7);

extern const float* get_BAR_F_PROB7();


extern float h_env_BAR_F_PROB7;

/** set_CLINIC_DUR
 * Sets the constant variable CLINIC_DUR on the device which can then be used in the agent functions.
 * @param h_CLINIC_DUR value to set the variable
 */
extern void set_CLINIC_DUR(float* h_CLINIC_DUR);

extern const float* get_CLINIC_DUR();


extern float h_env_CLINIC_DUR;

/** set_BAR_DUR
 * Sets the constant variable BAR_DUR on the device which can then be used in the agent functions.
 * @param h_BAR_DUR value to set the variable
 */
extern void set_BAR_DUR(float* h_BAR_DUR);

extern const float* get_BAR_DUR();


extern float h_env_BAR_DUR;

/** set_SCHOOL_DUR
 * Sets the constant variable SCHOOL_DUR on the device which can then be used in the agent functions.
 * @param h_SCHOOL_DUR value to set the variable
 */
extern void set_SCHOOL_DUR(float* h_SCHOOL_DUR);

extern const float* get_SCHOOL_DUR();


extern float h_env_SCHOOL_DUR;

/** set_VISITING_DUR
 * Sets the constant variable VISITING_DUR on the device which can then be used in the agent functions.
 * @param h_VISITING_DUR value to set the variable
 */
extern void set_VISITING_DUR(float* h_VISITING_DUR);

extern const float* get_VISITING_DUR();


extern float h_env_VISITING_DUR;

/** set_OUTPUT_ID
 * Sets the constant variable OUTPUT_ID on the device which can then be used in the agent functions.
 * @param h_OUTPUT_ID value to set the variable
 */
extern void set_OUTPUT_ID(unsigned int* h_OUTPUT_ID);

extern const unsigned int* get_OUTPUT_ID();


extern unsigned int h_env_OUTPUT_ID;


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

