
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

/* TBAssignment Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_TBAssignment_list* d_TBAssignments;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_TBAssignment_list* d_TBAssignments_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_TBAssignment_list* d_TBAssignments_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_TBAssignment_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_TBAssignment_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_TBAssignment_values;  /**< Agent sort identifiers value */

/* TBAssignment state variables */
xmachine_memory_TBAssignment_list* h_TBAssignments_tbdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_TBAssignment_list* d_TBAssignments_tbdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_TBAssignment_tbdefault_count;   /**< Agent population size counter */ 

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

/* TransportMembership Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_TransportMembership_list* d_TransportMemberships;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_TransportMembership_list* d_TransportMemberships_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_TransportMembership_list* d_TransportMemberships_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_TransportMembership_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_TransportMembership_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_TransportMembership_values;  /**< Agent sort identifiers value */

/* TransportMembership state variables */
xmachine_memory_TransportMembership_list* h_TransportMemberships_trmembershipdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_TransportMembership_list* d_TransportMemberships_trmembershipdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_TransportMembership_trmembershipdefault_count;   /**< Agent population size counter */ 

/* Clinic Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Clinic_list* d_Clinics;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Clinic_list* d_Clinics_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Clinic_list* d_Clinics_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Clinic_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Clinic_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Clinic_values;  /**< Agent sort identifiers value */

/* Clinic state variables */
xmachine_memory_Clinic_list* h_Clinics_cldefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Clinic_list* d_Clinics_cldefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Clinic_cldefault_count;   /**< Agent population size counter */ 

/* Workplace Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Workplace_list* d_Workplaces;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Workplace_list* d_Workplaces_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Workplace_list* d_Workplaces_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Workplace_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Workplace_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Workplace_values;  /**< Agent sort identifiers value */

/* Workplace state variables */
xmachine_memory_Workplace_list* h_Workplaces_wpdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Workplace_list* d_Workplaces_wpdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Workplace_wpdefault_count;   /**< Agent population size counter */ 

/* WorkplaceMembership Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_WorkplaceMembership_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_WorkplaceMembership_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_WorkplaceMembership_values;  /**< Agent sort identifiers value */

/* WorkplaceMembership state variables */
xmachine_memory_WorkplaceMembership_list* h_WorkplaceMemberships_wpmembershipdefault;      /**< Pointer to agent list (population) on host*/
xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_wpmembershipdefault;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_Persons_default_variable_id_data_iteration;
unsigned int h_Persons_default_variable_step_data_iteration;
unsigned int h_Persons_default_variable_householdtime_data_iteration;
unsigned int h_Persons_default_variable_churchtime_data_iteration;
unsigned int h_Persons_default_variable_transporttime_data_iteration;
unsigned int h_Persons_default_variable_clinictime_data_iteration;
unsigned int h_Persons_default_variable_workplacetime_data_iteration;
unsigned int h_Persons_default_variable_age_data_iteration;
unsigned int h_Persons_default_variable_gender_data_iteration;
unsigned int h_Persons_default_variable_householdsize_data_iteration;
unsigned int h_Persons_default_variable_churchfreq_data_iteration;
unsigned int h_Persons_default_variable_churchdur_data_iteration;
unsigned int h_Persons_default_variable_transportdur_data_iteration;
unsigned int h_Persons_default_variable_transportday1_data_iteration;
unsigned int h_Persons_default_variable_transportday2_data_iteration;
unsigned int h_Persons_default_variable_household_data_iteration;
unsigned int h_Persons_default_variable_church_data_iteration;
unsigned int h_Persons_default_variable_transport_data_iteration;
unsigned int h_Persons_default_variable_workplace_data_iteration;
unsigned int h_Persons_default_variable_busy_data_iteration;
unsigned int h_Persons_default_variable_startstep_data_iteration;
unsigned int h_Persons_default_variable_location_data_iteration;
unsigned int h_Persons_default_variable_locationid_data_iteration;
unsigned int h_Persons_default_variable_hiv_data_iteration;
unsigned int h_Persons_default_variable_art_data_iteration;
unsigned int h_Persons_default_variable_activetb_data_iteration;
unsigned int h_Persons_default_variable_artday_data_iteration;
unsigned int h_Persons_default_variable_p_data_iteration;
unsigned int h_Persons_default_variable_q_data_iteration;
unsigned int h_Persons_default_variable_infections_data_iteration;
unsigned int h_Persons_default_variable_lastinfected_data_iteration;
unsigned int h_Persons_default_variable_lastinfectedid_data_iteration;
unsigned int h_Persons_default_variable_time_step_data_iteration;
unsigned int h_Persons_default_variable_lambda_data_iteration;
unsigned int h_Persons_default_variable_timevisiting_data_iteration;
unsigned int h_Persons_s2_variable_id_data_iteration;
unsigned int h_Persons_s2_variable_step_data_iteration;
unsigned int h_Persons_s2_variable_householdtime_data_iteration;
unsigned int h_Persons_s2_variable_churchtime_data_iteration;
unsigned int h_Persons_s2_variable_transporttime_data_iteration;
unsigned int h_Persons_s2_variable_clinictime_data_iteration;
unsigned int h_Persons_s2_variable_workplacetime_data_iteration;
unsigned int h_Persons_s2_variable_age_data_iteration;
unsigned int h_Persons_s2_variable_gender_data_iteration;
unsigned int h_Persons_s2_variable_householdsize_data_iteration;
unsigned int h_Persons_s2_variable_churchfreq_data_iteration;
unsigned int h_Persons_s2_variable_churchdur_data_iteration;
unsigned int h_Persons_s2_variable_transportdur_data_iteration;
unsigned int h_Persons_s2_variable_transportday1_data_iteration;
unsigned int h_Persons_s2_variable_transportday2_data_iteration;
unsigned int h_Persons_s2_variable_household_data_iteration;
unsigned int h_Persons_s2_variable_church_data_iteration;
unsigned int h_Persons_s2_variable_transport_data_iteration;
unsigned int h_Persons_s2_variable_workplace_data_iteration;
unsigned int h_Persons_s2_variable_busy_data_iteration;
unsigned int h_Persons_s2_variable_startstep_data_iteration;
unsigned int h_Persons_s2_variable_location_data_iteration;
unsigned int h_Persons_s2_variable_locationid_data_iteration;
unsigned int h_Persons_s2_variable_hiv_data_iteration;
unsigned int h_Persons_s2_variable_art_data_iteration;
unsigned int h_Persons_s2_variable_activetb_data_iteration;
unsigned int h_Persons_s2_variable_artday_data_iteration;
unsigned int h_Persons_s2_variable_p_data_iteration;
unsigned int h_Persons_s2_variable_q_data_iteration;
unsigned int h_Persons_s2_variable_infections_data_iteration;
unsigned int h_Persons_s2_variable_lastinfected_data_iteration;
unsigned int h_Persons_s2_variable_lastinfectedid_data_iteration;
unsigned int h_Persons_s2_variable_time_step_data_iteration;
unsigned int h_Persons_s2_variable_lambda_data_iteration;
unsigned int h_Persons_s2_variable_timevisiting_data_iteration;
unsigned int h_TBAssignments_tbdefault_variable_id_data_iteration;
unsigned int h_Households_hhdefault_variable_id_data_iteration;
unsigned int h_Households_hhdefault_variable_lambda_data_iteration;
unsigned int h_Households_hhdefault_variable_active_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration;
unsigned int h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration;
unsigned int h_Churchs_chudefault_variable_id_data_iteration;
unsigned int h_Churchs_chudefault_variable_size_data_iteration;
unsigned int h_Churchs_chudefault_variable_lambda_data_iteration;
unsigned int h_Churchs_chudefault_variable_active_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration;
unsigned int h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration;
unsigned int h_Transports_trdefault_variable_id_data_iteration;
unsigned int h_Transports_trdefault_variable_lambda_data_iteration;
unsigned int h_Transports_trdefault_variable_active_data_iteration;
unsigned int h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration;
unsigned int h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration;
unsigned int h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration;
unsigned int h_Clinics_cldefault_variable_id_data_iteration;
unsigned int h_Clinics_cldefault_variable_lambda_data_iteration;
unsigned int h_Workplaces_wpdefault_variable_id_data_iteration;
unsigned int h_Workplaces_wpdefault_variable_lambda_data_iteration;
unsigned int h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration;
unsigned int h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration;


/* Message Memory */

/* tb_assignment Message variables */
xmachine_message_tb_assignment_list* h_tb_assignments;         /**< Pointer to message list on host*/
xmachine_message_tb_assignment_list* d_tb_assignments;         /**< Pointer to message list on device*/
xmachine_message_tb_assignment_list* d_tb_assignments_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_tb_assignment_count;         /**< message list counter*/
int h_message_tb_assignment_output_type;   /**< message output type (single or optional)*/

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

/* transport_membership Message variables */
xmachine_message_transport_membership_list* h_transport_memberships;         /**< Pointer to message list on host*/
xmachine_message_transport_membership_list* d_transport_memberships;         /**< Pointer to message list on device*/
xmachine_message_transport_membership_list* d_transport_memberships_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_transport_membership_count;         /**< message list counter*/
int h_message_transport_membership_output_type;   /**< message output type (single or optional)*/

/* workplace_membership Message variables */
xmachine_message_workplace_membership_list* h_workplace_memberships;         /**< Pointer to message list on host*/
xmachine_message_workplace_membership_list* d_workplace_memberships;         /**< Pointer to message list on device*/
xmachine_message_workplace_membership_list* d_workplace_memberships_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_workplace_membership_count;         /**< message list counter*/
int h_message_workplace_membership_output_type;   /**< message output type (single or optional)*/

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/

/* infection Message variables */
xmachine_message_infection_list* h_infections;         /**< Pointer to message list on host*/
xmachine_message_infection_list* d_infections;         /**< Pointer to message list on device*/
xmachine_message_infection_list* d_infections_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_infection_count;         /**< message list counter*/
int h_message_infection_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_Person;
size_t temp_scan_storage_bytes_Person;

void * d_temp_scan_storage_TBAssignment;
size_t temp_scan_storage_bytes_TBAssignment;

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

void * d_temp_scan_storage_TransportMembership;
size_t temp_scan_storage_bytes_TransportMembership;

void * d_temp_scan_storage_Clinic;
size_t temp_scan_storage_bytes_Clinic;

void * d_temp_scan_storage_Workplace;
size_t temp_scan_storage_bytes_Workplace;

void * d_temp_scan_storage_WorkplaceMembership;
size_t temp_scan_storage_bytes_WorkplaceMembership;


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

/** Person_updatelambda
 * Agent function prototype for updatelambda function of Person agent
 */
void Person_updatelambda(cudaStream_t &stream);

/** Person_infect
 * Agent function prototype for infect function of Person agent
 */
void Person_infect(cudaStream_t &stream);

/** Person_personhhinit
 * Agent function prototype for personhhinit function of Person agent
 */
void Person_personhhinit(cudaStream_t &stream);

/** Person_persontbinit
 * Agent function prototype for persontbinit function of Person agent
 */
void Person_persontbinit(cudaStream_t &stream);

/** Person_persontrinit
 * Agent function prototype for persontrinit function of Person agent
 */
void Person_persontrinit(cudaStream_t &stream);

/** Person_personwpinit
 * Agent function prototype for personwpinit function of Person agent
 */
void Person_personwpinit(cudaStream_t &stream);

/** TBAssignment_tbinit
 * Agent function prototype for tbinit function of TBAssignment agent
 */
void TBAssignment_tbinit(cudaStream_t &stream);

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

/** TransportMembership_trinit
 * Agent function prototype for trinit function of TransportMembership agent
 */
void TransportMembership_trinit(cudaStream_t &stream);

/** Clinic_clupdate
 * Agent function prototype for clupdate function of Clinic agent
 */
void Clinic_clupdate(cudaStream_t &stream);

/** Workplace_wpupdate
 * Agent function prototype for wpupdate function of Workplace agent
 */
void Workplace_wpupdate(cudaStream_t &stream);

/** WorkplaceMembership_wpinit
 * Agent function prototype for wpinit function of WorkplaceMembership agent
 */
void WorkplaceMembership_wpinit(cudaStream_t &stream);

  
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
    h_Persons_default_variable_householdtime_data_iteration = 0;
    h_Persons_default_variable_churchtime_data_iteration = 0;
    h_Persons_default_variable_transporttime_data_iteration = 0;
    h_Persons_default_variable_clinictime_data_iteration = 0;
    h_Persons_default_variable_workplacetime_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    h_Persons_default_variable_churchfreq_data_iteration = 0;
    h_Persons_default_variable_churchdur_data_iteration = 0;
    h_Persons_default_variable_transportdur_data_iteration = 0;
    h_Persons_default_variable_transportday1_data_iteration = 0;
    h_Persons_default_variable_transportday2_data_iteration = 0;
    h_Persons_default_variable_household_data_iteration = 0;
    h_Persons_default_variable_church_data_iteration = 0;
    h_Persons_default_variable_transport_data_iteration = 0;
    h_Persons_default_variable_workplace_data_iteration = 0;
    h_Persons_default_variable_busy_data_iteration = 0;
    h_Persons_default_variable_startstep_data_iteration = 0;
    h_Persons_default_variable_location_data_iteration = 0;
    h_Persons_default_variable_locationid_data_iteration = 0;
    h_Persons_default_variable_hiv_data_iteration = 0;
    h_Persons_default_variable_art_data_iteration = 0;
    h_Persons_default_variable_activetb_data_iteration = 0;
    h_Persons_default_variable_artday_data_iteration = 0;
    h_Persons_default_variable_p_data_iteration = 0;
    h_Persons_default_variable_q_data_iteration = 0;
    h_Persons_default_variable_infections_data_iteration = 0;
    h_Persons_default_variable_lastinfected_data_iteration = 0;
    h_Persons_default_variable_lastinfectedid_data_iteration = 0;
    h_Persons_default_variable_time_step_data_iteration = 0;
    h_Persons_default_variable_lambda_data_iteration = 0;
    h_Persons_default_variable_timevisiting_data_iteration = 0;
    h_Persons_s2_variable_id_data_iteration = 0;
    h_Persons_s2_variable_step_data_iteration = 0;
    h_Persons_s2_variable_householdtime_data_iteration = 0;
    h_Persons_s2_variable_churchtime_data_iteration = 0;
    h_Persons_s2_variable_transporttime_data_iteration = 0;
    h_Persons_s2_variable_clinictime_data_iteration = 0;
    h_Persons_s2_variable_workplacetime_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    h_Persons_s2_variable_churchfreq_data_iteration = 0;
    h_Persons_s2_variable_churchdur_data_iteration = 0;
    h_Persons_s2_variable_transportdur_data_iteration = 0;
    h_Persons_s2_variable_transportday1_data_iteration = 0;
    h_Persons_s2_variable_transportday2_data_iteration = 0;
    h_Persons_s2_variable_household_data_iteration = 0;
    h_Persons_s2_variable_church_data_iteration = 0;
    h_Persons_s2_variable_transport_data_iteration = 0;
    h_Persons_s2_variable_workplace_data_iteration = 0;
    h_Persons_s2_variable_busy_data_iteration = 0;
    h_Persons_s2_variable_startstep_data_iteration = 0;
    h_Persons_s2_variable_location_data_iteration = 0;
    h_Persons_s2_variable_locationid_data_iteration = 0;
    h_Persons_s2_variable_hiv_data_iteration = 0;
    h_Persons_s2_variable_art_data_iteration = 0;
    h_Persons_s2_variable_activetb_data_iteration = 0;
    h_Persons_s2_variable_artday_data_iteration = 0;
    h_Persons_s2_variable_p_data_iteration = 0;
    h_Persons_s2_variable_q_data_iteration = 0;
    h_Persons_s2_variable_infections_data_iteration = 0;
    h_Persons_s2_variable_lastinfected_data_iteration = 0;
    h_Persons_s2_variable_lastinfectedid_data_iteration = 0;
    h_Persons_s2_variable_time_step_data_iteration = 0;
    h_Persons_s2_variable_lambda_data_iteration = 0;
    h_Persons_s2_variable_timevisiting_data_iteration = 0;
    h_TBAssignments_tbdefault_variable_id_data_iteration = 0;
    h_Households_hhdefault_variable_id_data_iteration = 0;
    h_Households_hhdefault_variable_lambda_data_iteration = 0;
    h_Households_hhdefault_variable_active_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_household_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_person_id_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = 0;
    h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = 0;
    h_Churchs_chudefault_variable_id_data_iteration = 0;
    h_Churchs_chudefault_variable_size_data_iteration = 0;
    h_Churchs_chudefault_variable_lambda_data_iteration = 0;
    h_Churchs_chudefault_variable_active_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_church_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_household_id_data_iteration = 0;
    h_ChurchMemberships_chumembershipdefault_variable_churchdur_data_iteration = 0;
    h_Transports_trdefault_variable_id_data_iteration = 0;
    h_Transports_trdefault_variable_lambda_data_iteration = 0;
    h_Transports_trdefault_variable_active_data_iteration = 0;
    h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration = 0;
    h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration = 0;
    h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration = 0;
    h_Clinics_cldefault_variable_id_data_iteration = 0;
    h_Clinics_cldefault_variable_lambda_data_iteration = 0;
    h_Workplaces_wpdefault_variable_id_data_iteration = 0;
    h_Workplaces_wpdefault_variable_lambda_data_iteration = 0;
    h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration = 0;
    h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_Person_SoA_size = sizeof(xmachine_memory_Person_list);
	h_Persons_default = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);
	h_Persons_s2 = (xmachine_memory_Person_list*)malloc(xmachine_Person_SoA_size);
	int xmachine_TBAssignment_SoA_size = sizeof(xmachine_memory_TBAssignment_list);
	h_TBAssignments_tbdefault = (xmachine_memory_TBAssignment_list*)malloc(xmachine_TBAssignment_SoA_size);
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
	int xmachine_TransportMembership_SoA_size = sizeof(xmachine_memory_TransportMembership_list);
	h_TransportMemberships_trmembershipdefault = (xmachine_memory_TransportMembership_list*)malloc(xmachine_TransportMembership_SoA_size);
	int xmachine_Clinic_SoA_size = sizeof(xmachine_memory_Clinic_list);
	h_Clinics_cldefault = (xmachine_memory_Clinic_list*)malloc(xmachine_Clinic_SoA_size);
	int xmachine_Workplace_SoA_size = sizeof(xmachine_memory_Workplace_list);
	h_Workplaces_wpdefault = (xmachine_memory_Workplace_list*)malloc(xmachine_Workplace_SoA_size);
	int xmachine_WorkplaceMembership_SoA_size = sizeof(xmachine_memory_WorkplaceMembership_list);
	h_WorkplaceMemberships_wpmembershipdefault = (xmachine_memory_WorkplaceMembership_list*)malloc(xmachine_WorkplaceMembership_SoA_size);

	/* Message memory allocation (CPU) */
	int message_tb_assignment_SoA_size = sizeof(xmachine_message_tb_assignment_list);
	h_tb_assignments = (xmachine_message_tb_assignment_list*)malloc(message_tb_assignment_SoA_size);
	int message_household_membership_SoA_size = sizeof(xmachine_message_household_membership_list);
	h_household_memberships = (xmachine_message_household_membership_list*)malloc(message_household_membership_SoA_size);
	int message_church_membership_SoA_size = sizeof(xmachine_message_church_membership_list);
	h_church_memberships = (xmachine_message_church_membership_list*)malloc(message_church_membership_SoA_size);
	int message_transport_membership_SoA_size = sizeof(xmachine_message_transport_membership_list);
	h_transport_memberships = (xmachine_message_transport_membership_list*)malloc(message_transport_membership_SoA_size);
	int message_workplace_membership_SoA_size = sizeof(xmachine_message_workplace_membership_list);
	h_workplace_memberships = (xmachine_message_workplace_membership_list*)malloc(message_workplace_membership_SoA_size);
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);
	int message_infection_SoA_size = sizeof(xmachine_message_infection_list);
	h_infections = (xmachine_message_infection_list*)malloc(message_infection_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs
    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_Persons_default, &h_xmachine_memory_Person_default_count, h_TBAssignments_tbdefault, &h_xmachine_memory_TBAssignment_tbdefault_count, h_Households_hhdefault, &h_xmachine_memory_Household_hhdefault_count, h_HouseholdMemberships_hhmembershipdefault, &h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, h_Churchs_chudefault, &h_xmachine_memory_Church_chudefault_count, h_ChurchMemberships_chumembershipdefault, &h_xmachine_memory_ChurchMembership_chumembershipdefault_count, h_Transports_trdefault, &h_xmachine_memory_Transport_trdefault_count, h_TransportMemberships_trmembershipdefault, &h_xmachine_memory_TransportMembership_trmembershipdefault_count, h_Clinics_cldefault, &h_xmachine_memory_Clinic_cldefault_count, h_Workplaces_wpdefault, &h_xmachine_memory_Workplace_wpdefault_count, h_WorkplaceMemberships_wpmembershipdefault, &h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count);
	

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
    
	/* TBAssignment Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TBAssignments, xmachine_TBAssignment_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TBAssignments_swap, xmachine_TBAssignment_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TBAssignments_new, xmachine_TBAssignment_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TBAssignment_keys, xmachine_memory_TBAssignment_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TBAssignment_values, xmachine_memory_TBAssignment_MAX* sizeof(uint)));
	/* tbdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TBAssignments_tbdefault, xmachine_TBAssignment_SoA_size));
	gpuErrchk( cudaMemcpy( d_TBAssignments_tbdefault, h_TBAssignments_tbdefault, xmachine_TBAssignment_SoA_size, cudaMemcpyHostToDevice));
    
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
    
	/* TransportMembership Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TransportMemberships, xmachine_TransportMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TransportMemberships_swap, xmachine_TransportMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_TransportMemberships_new, xmachine_TransportMembership_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TransportMembership_keys, xmachine_memory_TransportMembership_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_TransportMembership_values, xmachine_memory_TransportMembership_MAX* sizeof(uint)));
	/* trmembershipdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_TransportMemberships_trmembershipdefault, xmachine_TransportMembership_SoA_size));
	gpuErrchk( cudaMemcpy( d_TransportMemberships_trmembershipdefault, h_TransportMemberships_trmembershipdefault, xmachine_TransportMembership_SoA_size, cudaMemcpyHostToDevice));
    
	/* Clinic Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Clinics, xmachine_Clinic_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Clinics_swap, xmachine_Clinic_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Clinics_new, xmachine_Clinic_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Clinic_keys, xmachine_memory_Clinic_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Clinic_values, xmachine_memory_Clinic_MAX* sizeof(uint)));
	/* cldefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Clinics_cldefault, xmachine_Clinic_SoA_size));
	gpuErrchk( cudaMemcpy( d_Clinics_cldefault, h_Clinics_cldefault, xmachine_Clinic_SoA_size, cudaMemcpyHostToDevice));
    
	/* Workplace Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Workplaces, xmachine_Workplace_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Workplaces_swap, xmachine_Workplace_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Workplaces_new, xmachine_Workplace_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Workplace_keys, xmachine_memory_Workplace_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Workplace_values, xmachine_memory_Workplace_MAX* sizeof(uint)));
	/* wpdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Workplaces_wpdefault, xmachine_Workplace_SoA_size));
	gpuErrchk( cudaMemcpy( d_Workplaces_wpdefault, h_Workplaces_wpdefault, xmachine_Workplace_SoA_size, cudaMemcpyHostToDevice));
    
	/* WorkplaceMembership Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_WorkplaceMemberships, xmachine_WorkplaceMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_WorkplaceMemberships_swap, xmachine_WorkplaceMembership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_WorkplaceMemberships_new, xmachine_WorkplaceMembership_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_WorkplaceMembership_keys, xmachine_memory_WorkplaceMembership_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_WorkplaceMembership_values, xmachine_memory_WorkplaceMembership_MAX* sizeof(uint)));
	/* wpmembershipdefault memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_WorkplaceMemberships_wpmembershipdefault, xmachine_WorkplaceMembership_SoA_size));
	gpuErrchk( cudaMemcpy( d_WorkplaceMemberships_wpmembershipdefault, h_WorkplaceMemberships_wpmembershipdefault, xmachine_WorkplaceMembership_SoA_size, cudaMemcpyHostToDevice));
    
	/* tb_assignment Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_tb_assignments, message_tb_assignment_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_tb_assignments_swap, message_tb_assignment_SoA_size));
	gpuErrchk( cudaMemcpy( d_tb_assignments, h_tb_assignments, message_tb_assignment_SoA_size, cudaMemcpyHostToDevice));
	
	/* household_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_household_memberships, message_household_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_household_memberships_swap, message_household_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_household_memberships, h_household_memberships, message_household_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* church_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_church_memberships, message_church_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_church_memberships_swap, message_church_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_church_memberships, h_church_memberships, message_church_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* transport_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_transport_memberships, message_transport_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_transport_memberships_swap, message_transport_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_transport_memberships, h_transport_memberships, message_transport_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* workplace_membership Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_workplace_memberships, message_workplace_membership_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_workplace_memberships_swap, message_workplace_membership_SoA_size));
	gpuErrchk( cudaMemcpy( d_workplace_memberships, h_workplace_memberships, message_workplace_membership_SoA_size, cudaMemcpyHostToDevice));
	
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
	
	/* infection Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_infections, message_infection_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_infections_swap, message_infection_SoA_size));
	gpuErrchk( cudaMemcpy( d_infections, h_infections, message_infection_SoA_size, cudaMemcpyHostToDevice));
		
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
    
    d_temp_scan_storage_TBAssignment = nullptr;
    temp_scan_storage_bytes_TBAssignment = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TBAssignment, 
        temp_scan_storage_bytes_TBAssignment, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_TBAssignment_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_TBAssignment, temp_scan_storage_bytes_TBAssignment));
    
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
    
    d_temp_scan_storage_TransportMembership = nullptr;
    temp_scan_storage_bytes_TransportMembership = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TransportMembership, 
        temp_scan_storage_bytes_TransportMembership, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_TransportMembership_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_TransportMembership, temp_scan_storage_bytes_TransportMembership));
    
    d_temp_scan_storage_Clinic = nullptr;
    temp_scan_storage_bytes_Clinic = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Clinic, 
        temp_scan_storage_bytes_Clinic, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Clinic_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Clinic, temp_scan_storage_bytes_Clinic));
    
    d_temp_scan_storage_Workplace = nullptr;
    temp_scan_storage_bytes_Workplace = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Workplace, 
        temp_scan_storage_bytes_Workplace, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Workplace_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Workplace, temp_scan_storage_bytes_Workplace));
    
    d_temp_scan_storage_WorkplaceMembership = nullptr;
    temp_scan_storage_bytes_WorkplaceMembership = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_WorkplaceMembership, 
        temp_scan_storage_bytes_WorkplaceMembership, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_WorkplaceMembership_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_WorkplaceMembership, temp_scan_storage_bytes_WorkplaceMembership));
    

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
	
		printf("Init agent_TBAssignment_tbdefault_count: %u\n",get_agent_TBAssignment_tbdefault_count());
	
		printf("Init agent_Household_hhdefault_count: %u\n",get_agent_Household_hhdefault_count());
	
		printf("Init agent_HouseholdMembership_hhmembershipdefault_count: %u\n",get_agent_HouseholdMembership_hhmembershipdefault_count());
	
		printf("Init agent_Church_chudefault_count: %u\n",get_agent_Church_chudefault_count());
	
		printf("Init agent_ChurchMembership_chumembershipdefault_count: %u\n",get_agent_ChurchMembership_chumembershipdefault_count());
	
		printf("Init agent_Transport_trdefault_count: %u\n",get_agent_Transport_trdefault_count());
	
		printf("Init agent_TransportMembership_trmembershipdefault_count: %u\n",get_agent_TransportMembership_trmembershipdefault_count());
	
		printf("Init agent_Clinic_cldefault_count: %u\n",get_agent_Clinic_cldefault_count());
	
		printf("Init agent_Workplace_wpdefault_count: %u\n",get_agent_Workplace_wpdefault_count());
	
		printf("Init agent_WorkplaceMembership_wpmembershipdefault_count: %u\n",get_agent_WorkplaceMembership_wpmembershipdefault_count());
	
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

void sort_TBAssignments_tbdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TBAssignment_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_TBAssignment_tbdefault_count); 
	gridSize = (h_xmachine_memory_TBAssignment_tbdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_TBAssignment_keys, d_xmachine_memory_TBAssignment_values, d_TBAssignments_tbdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_TBAssignment_keys),  thrust::device_pointer_cast(d_xmachine_memory_TBAssignment_keys) + h_xmachine_memory_TBAssignment_tbdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_TBAssignment_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_TBAssignment_agents, no_sm, h_xmachine_memory_TBAssignment_tbdefault_count); 
	gridSize = (h_xmachine_memory_TBAssignment_tbdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_TBAssignment_agents<<<gridSize, blockSize>>>(d_xmachine_memory_TBAssignment_values, d_TBAssignments_tbdefault, d_TBAssignments_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_TBAssignment_list* d_TBAssignments_temp = d_TBAssignments_tbdefault;
	d_TBAssignments_tbdefault = d_TBAssignments_swap;
	d_TBAssignments_swap = d_TBAssignments_temp;	
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

void sort_TransportMemberships_trmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_TransportMembership_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_TransportMembership_trmembershipdefault_count); 
	gridSize = (h_xmachine_memory_TransportMembership_trmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_TransportMembership_keys, d_xmachine_memory_TransportMembership_values, d_TransportMemberships_trmembershipdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_TransportMembership_keys),  thrust::device_pointer_cast(d_xmachine_memory_TransportMembership_keys) + h_xmachine_memory_TransportMembership_trmembershipdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_TransportMembership_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_TransportMembership_agents, no_sm, h_xmachine_memory_TransportMembership_trmembershipdefault_count); 
	gridSize = (h_xmachine_memory_TransportMembership_trmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_TransportMembership_agents<<<gridSize, blockSize>>>(d_xmachine_memory_TransportMembership_values, d_TransportMemberships_trmembershipdefault, d_TransportMemberships_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_TransportMembership_list* d_TransportMemberships_temp = d_TransportMemberships_trmembershipdefault;
	d_TransportMemberships_trmembershipdefault = d_TransportMemberships_swap;
	d_TransportMemberships_swap = d_TransportMemberships_temp;	
}

void sort_Clinics_cldefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Clinic_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Clinic_cldefault_count); 
	gridSize = (h_xmachine_memory_Clinic_cldefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Clinic_keys, d_xmachine_memory_Clinic_values, d_Clinics_cldefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Clinic_keys),  thrust::device_pointer_cast(d_xmachine_memory_Clinic_keys) + h_xmachine_memory_Clinic_cldefault_count,  thrust::device_pointer_cast(d_xmachine_memory_Clinic_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Clinic_agents, no_sm, h_xmachine_memory_Clinic_cldefault_count); 
	gridSize = (h_xmachine_memory_Clinic_cldefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Clinic_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Clinic_values, d_Clinics_cldefault, d_Clinics_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Clinic_list* d_Clinics_temp = d_Clinics_cldefault;
	d_Clinics_cldefault = d_Clinics_swap;
	d_Clinics_swap = d_Clinics_temp;	
}

void sort_Workplaces_wpdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Workplace_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Workplace_wpdefault_count); 
	gridSize = (h_xmachine_memory_Workplace_wpdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Workplace_keys, d_xmachine_memory_Workplace_values, d_Workplaces_wpdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Workplace_keys),  thrust::device_pointer_cast(d_xmachine_memory_Workplace_keys) + h_xmachine_memory_Workplace_wpdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_Workplace_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Workplace_agents, no_sm, h_xmachine_memory_Workplace_wpdefault_count); 
	gridSize = (h_xmachine_memory_Workplace_wpdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Workplace_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Workplace_values, d_Workplaces_wpdefault, d_Workplaces_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Workplace_list* d_Workplaces_temp = d_Workplaces_wpdefault;
	d_Workplaces_wpdefault = d_Workplaces_swap;
	d_Workplaces_swap = d_Workplaces_temp;	
}

void sort_WorkplaceMemberships_wpmembershipdefault(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_WorkplaceMembership_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count); 
	gridSize = (h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_WorkplaceMembership_keys, d_xmachine_memory_WorkplaceMembership_values, d_WorkplaceMemberships_wpmembershipdefault);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_WorkplaceMembership_keys),  thrust::device_pointer_cast(d_xmachine_memory_WorkplaceMembership_keys) + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count,  thrust::device_pointer_cast(d_xmachine_memory_WorkplaceMembership_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_WorkplaceMembership_agents, no_sm, h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count); 
	gridSize = (h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_WorkplaceMembership_agents<<<gridSize, blockSize>>>(d_xmachine_memory_WorkplaceMembership_values, d_WorkplaceMemberships_wpmembershipdefault, d_WorkplaceMemberships_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_temp = d_WorkplaceMemberships_wpmembershipdefault;
	d_WorkplaceMemberships_wpmembershipdefault = d_WorkplaceMemberships_swap;
	d_WorkplaceMemberships_swap = d_WorkplaceMemberships_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

    customOutputFunction();
    PROFILE_PUSH_RANGE("customOutputFunction");
	PROFILE_POP_RANGE();

#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: customOutputFunction = %f (ms)\n", instrument_milliseconds);
#endif
	
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
	
	/* TBAssignment Agent variables */
	gpuErrchk(cudaFree(d_TBAssignments));
	gpuErrchk(cudaFree(d_TBAssignments_swap));
	gpuErrchk(cudaFree(d_TBAssignments_new));
	
	free( h_TBAssignments_tbdefault);
	gpuErrchk(cudaFree(d_TBAssignments_tbdefault));
	
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
	
	/* TransportMembership Agent variables */
	gpuErrchk(cudaFree(d_TransportMemberships));
	gpuErrchk(cudaFree(d_TransportMemberships_swap));
	gpuErrchk(cudaFree(d_TransportMemberships_new));
	
	free( h_TransportMemberships_trmembershipdefault);
	gpuErrchk(cudaFree(d_TransportMemberships_trmembershipdefault));
	
	/* Clinic Agent variables */
	gpuErrchk(cudaFree(d_Clinics));
	gpuErrchk(cudaFree(d_Clinics_swap));
	gpuErrchk(cudaFree(d_Clinics_new));
	
	free( h_Clinics_cldefault);
	gpuErrchk(cudaFree(d_Clinics_cldefault));
	
	/* Workplace Agent variables */
	gpuErrchk(cudaFree(d_Workplaces));
	gpuErrchk(cudaFree(d_Workplaces_swap));
	gpuErrchk(cudaFree(d_Workplaces_new));
	
	free( h_Workplaces_wpdefault);
	gpuErrchk(cudaFree(d_Workplaces_wpdefault));
	
	/* WorkplaceMembership Agent variables */
	gpuErrchk(cudaFree(d_WorkplaceMemberships));
	gpuErrchk(cudaFree(d_WorkplaceMemberships_swap));
	gpuErrchk(cudaFree(d_WorkplaceMemberships_new));
	
	free( h_WorkplaceMemberships_wpmembershipdefault);
	gpuErrchk(cudaFree(d_WorkplaceMemberships_wpmembershipdefault));
	

	/* Message data free */
	
	/* tb_assignment Message variables */
	free( h_tb_assignments);
	gpuErrchk(cudaFree(d_tb_assignments));
	gpuErrchk(cudaFree(d_tb_assignments_swap));
	
	/* household_membership Message variables */
	free( h_household_memberships);
	gpuErrchk(cudaFree(d_household_memberships));
	gpuErrchk(cudaFree(d_household_memberships_swap));
	
	/* church_membership Message variables */
	free( h_church_memberships);
	gpuErrchk(cudaFree(d_church_memberships));
	gpuErrchk(cudaFree(d_church_memberships_swap));
	
	/* transport_membership Message variables */
	free( h_transport_memberships);
	gpuErrchk(cudaFree(d_transport_memberships));
	gpuErrchk(cudaFree(d_transport_memberships_swap));
	
	/* workplace_membership Message variables */
	free( h_workplace_memberships);
	gpuErrchk(cudaFree(d_workplace_memberships));
	gpuErrchk(cudaFree(d_workplace_memberships_swap));
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	
	/* infection Message variables */
	free( h_infections);
	gpuErrchk(cudaFree(d_infections));
	gpuErrchk(cudaFree(d_infections_swap));
	

    /* Free temporary CUB memory */
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Person));
    d_temp_scan_storage_Person = nullptr;
    temp_scan_storage_bytes_Person = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_TBAssignment));
    d_temp_scan_storage_TBAssignment = nullptr;
    temp_scan_storage_bytes_TBAssignment = 0;
    
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
    
    gpuErrchk(cudaFree(d_temp_scan_storage_TransportMembership));
    d_temp_scan_storage_TransportMembership = nullptr;
    temp_scan_storage_bytes_TransportMembership = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Clinic));
    d_temp_scan_storage_Clinic = nullptr;
    temp_scan_storage_bytes_Clinic = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_Workplace));
    d_temp_scan_storage_Workplace = nullptr;
    temp_scan_storage_bytes_Workplace = 0;
    
    gpuErrchk(cudaFree(d_temp_scan_storage_WorkplaceMembership));
    d_temp_scan_storage_WorkplaceMembership = nullptr;
    temp_scan_storage_bytes_WorkplaceMembership = 0;
    
  
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
	h_message_tb_assignment_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_tb_assignment_count, &h_message_tb_assignment_count, sizeof(int)));
	
	h_message_household_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_household_membership_count, &h_message_household_membership_count, sizeof(int)));
	
	h_message_church_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_church_membership_count, &h_message_church_membership_count, sizeof(int)));
	
	h_message_transport_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_transport_membership_count, &h_message_transport_membership_count, sizeof(int)));
	
	h_message_workplace_membership_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_workplace_membership_count, &h_message_workplace_membership_count, sizeof(int)));
	
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	
	h_message_infection_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));
	

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
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Household_hhupdate");
	Household_hhupdate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Household_hhupdate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Church_chuupdate");
	Church_chuupdate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Church_chuupdate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Transport_trupdate");
	Transport_trupdate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Transport_trupdate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Clinic_clupdate");
	Clinic_clupdate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Clinic_clupdate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Workplace_wpupdate");
	Workplace_wpupdate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Workplace_wpupdate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_updatelambda");
	Person_updatelambda(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_updatelambda = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 8*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_infect");
	Person_infect(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_infect = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 9*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TBAssignment_tbinit");
	TBAssignment_tbinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TBAssignment_tbinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 10*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("WorkplaceMembership_wpinit");
	WorkplaceMembership_wpinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: WorkplaceMembership_wpinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 11*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("TransportMembership_trinit");
	TransportMembership_trinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: TransportMembership_trinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 12*/
	
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
  
	/* Layer 13*/
	
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
  
	/* Layer 14*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_personwpinit");
	Person_personwpinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_personwpinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 15*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_persontrinit");
	Person_persontrinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_persontrinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 16*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_persontbinit");
	Person_persontbinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_persontbinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 17*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Person_personhhinit");
	Person_personhhinit(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Person_personhhinit = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_Person_default_count: %u\n",get_agent_Person_default_count());
	
		printf("agent_Person_s2_count: %u\n",get_agent_Person_s2_count());
	
		printf("agent_TBAssignment_tbdefault_count: %u\n",get_agent_TBAssignment_tbdefault_count());
	
		printf("agent_Household_hhdefault_count: %u\n",get_agent_Household_hhdefault_count());
	
		printf("agent_HouseholdMembership_hhmembershipdefault_count: %u\n",get_agent_HouseholdMembership_hhmembershipdefault_count());
	
		printf("agent_Church_chudefault_count: %u\n",get_agent_Church_chudefault_count());
	
		printf("agent_ChurchMembership_chumembershipdefault_count: %u\n",get_agent_ChurchMembership_chumembershipdefault_count());
	
		printf("agent_Transport_trdefault_count: %u\n",get_agent_Transport_trdefault_count());
	
		printf("agent_TransportMembership_trmembershipdefault_count: %u\n",get_agent_TransportMembership_trmembershipdefault_count());
	
		printf("agent_Clinic_cldefault_count: %u\n",get_agent_Clinic_cldefault_count());
	
		printf("agent_Workplace_wpdefault_count: %u\n",get_agent_Workplace_wpdefault_count());
	
		printf("agent_WorkplaceMembership_wpmembershipdefault_count: %u\n",get_agent_WorkplaceMembership_wpmembershipdefault_count());
	
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
float h_env_HIV_PREVALENCE;
float h_env_ART_COVERAGE;
float h_env_RR_HIV;
float h_env_RR_ART;
float h_env_TB_PREVALENCE;
float h_env_DEFAULT_P;
float h_env_DEFAULT_Q;
float h_env_TRANSPORT_A;
float h_env_CHURCH_A;
float h_env_CLINIC_A;
float h_env_HOUSEHOLD_A;
float h_env_TRANSPORT_V;
float h_env_HOUSEHOLD_V;
float h_env_CLINIC_V;
float h_env_CHURCH_V_MULTIPLIER;
float h_env_WORKPLACE_BETA0;
float h_env_WORKPLACE_BETAA;
float h_env_WORKPLACE_BETAS;
float h_env_WORKPLACE_BETAAS;
float h_env_WORKPLACE_A;
unsigned int h_env_WORKPLACE_DUR;
unsigned int h_env_WORKPLACE_SIZE;
float h_env_WORKPLACE_V;
unsigned int h_env_HOUSEHOLDS;
float h_env_RR_AS_F_46;
float h_env_RR_AS_F_26;
float h_env_RR_AS_F_18;
float h_env_RR_AS_M_46;
float h_env_RR_AS_M_26;
float h_env_RR_AS_M_18;


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



//constant setter
void set_HIV_PREVALENCE(float* h_HIV_PREVALENCE){
    gpuErrchk(cudaMemcpyToSymbol(HIV_PREVALENCE, h_HIV_PREVALENCE, sizeof(float)));
    memcpy(&h_env_HIV_PREVALENCE, h_HIV_PREVALENCE,sizeof(float));
}

//constant getter
const float* get_HIV_PREVALENCE(){
    return &h_env_HIV_PREVALENCE;
}



//constant setter
void set_ART_COVERAGE(float* h_ART_COVERAGE){
    gpuErrchk(cudaMemcpyToSymbol(ART_COVERAGE, h_ART_COVERAGE, sizeof(float)));
    memcpy(&h_env_ART_COVERAGE, h_ART_COVERAGE,sizeof(float));
}

//constant getter
const float* get_ART_COVERAGE(){
    return &h_env_ART_COVERAGE;
}



//constant setter
void set_RR_HIV(float* h_RR_HIV){
    gpuErrchk(cudaMemcpyToSymbol(RR_HIV, h_RR_HIV, sizeof(float)));
    memcpy(&h_env_RR_HIV, h_RR_HIV,sizeof(float));
}

//constant getter
const float* get_RR_HIV(){
    return &h_env_RR_HIV;
}



//constant setter
void set_RR_ART(float* h_RR_ART){
    gpuErrchk(cudaMemcpyToSymbol(RR_ART, h_RR_ART, sizeof(float)));
    memcpy(&h_env_RR_ART, h_RR_ART,sizeof(float));
}

//constant getter
const float* get_RR_ART(){
    return &h_env_RR_ART;
}



//constant setter
void set_TB_PREVALENCE(float* h_TB_PREVALENCE){
    gpuErrchk(cudaMemcpyToSymbol(TB_PREVALENCE, h_TB_PREVALENCE, sizeof(float)));
    memcpy(&h_env_TB_PREVALENCE, h_TB_PREVALENCE,sizeof(float));
}

//constant getter
const float* get_TB_PREVALENCE(){
    return &h_env_TB_PREVALENCE;
}



//constant setter
void set_DEFAULT_P(float* h_DEFAULT_P){
    gpuErrchk(cudaMemcpyToSymbol(DEFAULT_P, h_DEFAULT_P, sizeof(float)));
    memcpy(&h_env_DEFAULT_P, h_DEFAULT_P,sizeof(float));
}

//constant getter
const float* get_DEFAULT_P(){
    return &h_env_DEFAULT_P;
}



//constant setter
void set_DEFAULT_Q(float* h_DEFAULT_Q){
    gpuErrchk(cudaMemcpyToSymbol(DEFAULT_Q, h_DEFAULT_Q, sizeof(float)));
    memcpy(&h_env_DEFAULT_Q, h_DEFAULT_Q,sizeof(float));
}

//constant getter
const float* get_DEFAULT_Q(){
    return &h_env_DEFAULT_Q;
}



//constant setter
void set_TRANSPORT_A(float* h_TRANSPORT_A){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_A, h_TRANSPORT_A, sizeof(float)));
    memcpy(&h_env_TRANSPORT_A, h_TRANSPORT_A,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_A(){
    return &h_env_TRANSPORT_A;
}



//constant setter
void set_CHURCH_A(float* h_CHURCH_A){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_A, h_CHURCH_A, sizeof(float)));
    memcpy(&h_env_CHURCH_A, h_CHURCH_A,sizeof(float));
}

//constant getter
const float* get_CHURCH_A(){
    return &h_env_CHURCH_A;
}



//constant setter
void set_CLINIC_A(float* h_CLINIC_A){
    gpuErrchk(cudaMemcpyToSymbol(CLINIC_A, h_CLINIC_A, sizeof(float)));
    memcpy(&h_env_CLINIC_A, h_CLINIC_A,sizeof(float));
}

//constant getter
const float* get_CLINIC_A(){
    return &h_env_CLINIC_A;
}



//constant setter
void set_HOUSEHOLD_A(float* h_HOUSEHOLD_A){
    gpuErrchk(cudaMemcpyToSymbol(HOUSEHOLD_A, h_HOUSEHOLD_A, sizeof(float)));
    memcpy(&h_env_HOUSEHOLD_A, h_HOUSEHOLD_A,sizeof(float));
}

//constant getter
const float* get_HOUSEHOLD_A(){
    return &h_env_HOUSEHOLD_A;
}



//constant setter
void set_TRANSPORT_V(float* h_TRANSPORT_V){
    gpuErrchk(cudaMemcpyToSymbol(TRANSPORT_V, h_TRANSPORT_V, sizeof(float)));
    memcpy(&h_env_TRANSPORT_V, h_TRANSPORT_V,sizeof(float));
}

//constant getter
const float* get_TRANSPORT_V(){
    return &h_env_TRANSPORT_V;
}



//constant setter
void set_HOUSEHOLD_V(float* h_HOUSEHOLD_V){
    gpuErrchk(cudaMemcpyToSymbol(HOUSEHOLD_V, h_HOUSEHOLD_V, sizeof(float)));
    memcpy(&h_env_HOUSEHOLD_V, h_HOUSEHOLD_V,sizeof(float));
}

//constant getter
const float* get_HOUSEHOLD_V(){
    return &h_env_HOUSEHOLD_V;
}



//constant setter
void set_CLINIC_V(float* h_CLINIC_V){
    gpuErrchk(cudaMemcpyToSymbol(CLINIC_V, h_CLINIC_V, sizeof(float)));
    memcpy(&h_env_CLINIC_V, h_CLINIC_V,sizeof(float));
}

//constant getter
const float* get_CLINIC_V(){
    return &h_env_CLINIC_V;
}



//constant setter
void set_CHURCH_V_MULTIPLIER(float* h_CHURCH_V_MULTIPLIER){
    gpuErrchk(cudaMemcpyToSymbol(CHURCH_V_MULTIPLIER, h_CHURCH_V_MULTIPLIER, sizeof(float)));
    memcpy(&h_env_CHURCH_V_MULTIPLIER, h_CHURCH_V_MULTIPLIER,sizeof(float));
}

//constant getter
const float* get_CHURCH_V_MULTIPLIER(){
    return &h_env_CHURCH_V_MULTIPLIER;
}



//constant setter
void set_WORKPLACE_BETA0(float* h_WORKPLACE_BETA0){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_BETA0, h_WORKPLACE_BETA0, sizeof(float)));
    memcpy(&h_env_WORKPLACE_BETA0, h_WORKPLACE_BETA0,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_BETA0(){
    return &h_env_WORKPLACE_BETA0;
}



//constant setter
void set_WORKPLACE_BETAA(float* h_WORKPLACE_BETAA){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_BETAA, h_WORKPLACE_BETAA, sizeof(float)));
    memcpy(&h_env_WORKPLACE_BETAA, h_WORKPLACE_BETAA,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_BETAA(){
    return &h_env_WORKPLACE_BETAA;
}



//constant setter
void set_WORKPLACE_BETAS(float* h_WORKPLACE_BETAS){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_BETAS, h_WORKPLACE_BETAS, sizeof(float)));
    memcpy(&h_env_WORKPLACE_BETAS, h_WORKPLACE_BETAS,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_BETAS(){
    return &h_env_WORKPLACE_BETAS;
}



//constant setter
void set_WORKPLACE_BETAAS(float* h_WORKPLACE_BETAAS){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_BETAAS, h_WORKPLACE_BETAAS, sizeof(float)));
    memcpy(&h_env_WORKPLACE_BETAAS, h_WORKPLACE_BETAAS,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_BETAAS(){
    return &h_env_WORKPLACE_BETAAS;
}



//constant setter
void set_WORKPLACE_A(float* h_WORKPLACE_A){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_A, h_WORKPLACE_A, sizeof(float)));
    memcpy(&h_env_WORKPLACE_A, h_WORKPLACE_A,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_A(){
    return &h_env_WORKPLACE_A;
}



//constant setter
void set_WORKPLACE_DUR(unsigned int* h_WORKPLACE_DUR){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_DUR, h_WORKPLACE_DUR, sizeof(unsigned int)));
    memcpy(&h_env_WORKPLACE_DUR, h_WORKPLACE_DUR,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_WORKPLACE_DUR(){
    return &h_env_WORKPLACE_DUR;
}



//constant setter
void set_WORKPLACE_SIZE(unsigned int* h_WORKPLACE_SIZE){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_SIZE, h_WORKPLACE_SIZE, sizeof(unsigned int)));
    memcpy(&h_env_WORKPLACE_SIZE, h_WORKPLACE_SIZE,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_WORKPLACE_SIZE(){
    return &h_env_WORKPLACE_SIZE;
}



//constant setter
void set_WORKPLACE_V(float* h_WORKPLACE_V){
    gpuErrchk(cudaMemcpyToSymbol(WORKPLACE_V, h_WORKPLACE_V, sizeof(float)));
    memcpy(&h_env_WORKPLACE_V, h_WORKPLACE_V,sizeof(float));
}

//constant getter
const float* get_WORKPLACE_V(){
    return &h_env_WORKPLACE_V;
}



//constant setter
void set_HOUSEHOLDS(unsigned int* h_HOUSEHOLDS){
    gpuErrchk(cudaMemcpyToSymbol(HOUSEHOLDS, h_HOUSEHOLDS, sizeof(unsigned int)));
    memcpy(&h_env_HOUSEHOLDS, h_HOUSEHOLDS,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_HOUSEHOLDS(){
    return &h_env_HOUSEHOLDS;
}



//constant setter
void set_RR_AS_F_46(float* h_RR_AS_F_46){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_F_46, h_RR_AS_F_46, sizeof(float)));
    memcpy(&h_env_RR_AS_F_46, h_RR_AS_F_46,sizeof(float));
}

//constant getter
const float* get_RR_AS_F_46(){
    return &h_env_RR_AS_F_46;
}



//constant setter
void set_RR_AS_F_26(float* h_RR_AS_F_26){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_F_26, h_RR_AS_F_26, sizeof(float)));
    memcpy(&h_env_RR_AS_F_26, h_RR_AS_F_26,sizeof(float));
}

//constant getter
const float* get_RR_AS_F_26(){
    return &h_env_RR_AS_F_26;
}



//constant setter
void set_RR_AS_F_18(float* h_RR_AS_F_18){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_F_18, h_RR_AS_F_18, sizeof(float)));
    memcpy(&h_env_RR_AS_F_18, h_RR_AS_F_18,sizeof(float));
}

//constant getter
const float* get_RR_AS_F_18(){
    return &h_env_RR_AS_F_18;
}



//constant setter
void set_RR_AS_M_46(float* h_RR_AS_M_46){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_M_46, h_RR_AS_M_46, sizeof(float)));
    memcpy(&h_env_RR_AS_M_46, h_RR_AS_M_46,sizeof(float));
}

//constant getter
const float* get_RR_AS_M_46(){
    return &h_env_RR_AS_M_46;
}



//constant setter
void set_RR_AS_M_26(float* h_RR_AS_M_26){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_M_26, h_RR_AS_M_26, sizeof(float)));
    memcpy(&h_env_RR_AS_M_26, h_RR_AS_M_26,sizeof(float));
}

//constant getter
const float* get_RR_AS_M_26(){
    return &h_env_RR_AS_M_26;
}



//constant setter
void set_RR_AS_M_18(float* h_RR_AS_M_18){
    gpuErrchk(cudaMemcpyToSymbol(RR_AS_M_18, h_RR_AS_M_18, sizeof(float)));
    memcpy(&h_env_RR_AS_M_18, h_RR_AS_M_18,sizeof(float));
}

//constant getter
const float* get_RR_AS_M_18(){
    return &h_env_RR_AS_M_18;
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

    
int get_agent_TBAssignment_MAX_count(){
    return xmachine_memory_TBAssignment_MAX;
}


int get_agent_TBAssignment_tbdefault_count(){
	//continuous agent
	return h_xmachine_memory_TBAssignment_tbdefault_count;
	
}

xmachine_memory_TBAssignment_list* get_device_TBAssignment_tbdefault_agents(){
	return d_TBAssignments_tbdefault;
}

xmachine_memory_TBAssignment_list* get_host_TBAssignment_tbdefault_agents(){
	return h_TBAssignments_tbdefault;
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

    
int get_agent_TransportMembership_MAX_count(){
    return xmachine_memory_TransportMembership_MAX;
}


int get_agent_TransportMembership_trmembershipdefault_count(){
	//continuous agent
	return h_xmachine_memory_TransportMembership_trmembershipdefault_count;
	
}

xmachine_memory_TransportMembership_list* get_device_TransportMembership_trmembershipdefault_agents(){
	return d_TransportMemberships_trmembershipdefault;
}

xmachine_memory_TransportMembership_list* get_host_TransportMembership_trmembershipdefault_agents(){
	return h_TransportMemberships_trmembershipdefault;
}

    
int get_agent_Clinic_MAX_count(){
    return xmachine_memory_Clinic_MAX;
}


int get_agent_Clinic_cldefault_count(){
	//continuous agent
	return h_xmachine_memory_Clinic_cldefault_count;
	
}

xmachine_memory_Clinic_list* get_device_Clinic_cldefault_agents(){
	return d_Clinics_cldefault;
}

xmachine_memory_Clinic_list* get_host_Clinic_cldefault_agents(){
	return h_Clinics_cldefault;
}

    
int get_agent_Workplace_MAX_count(){
    return xmachine_memory_Workplace_MAX;
}


int get_agent_Workplace_wpdefault_count(){
	//continuous agent
	return h_xmachine_memory_Workplace_wpdefault_count;
	
}

xmachine_memory_Workplace_list* get_device_Workplace_wpdefault_agents(){
	return d_Workplaces_wpdefault;
}

xmachine_memory_Workplace_list* get_host_Workplace_wpdefault_agents(){
	return h_Workplaces_wpdefault;
}

    
int get_agent_WorkplaceMembership_MAX_count(){
    return xmachine_memory_WorkplaceMembership_MAX;
}


int get_agent_WorkplaceMembership_wpmembershipdefault_count(){
	//continuous agent
	return h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count;
	
}

xmachine_memory_WorkplaceMembership_list* get_device_WorkplaceMembership_wpmembershipdefault_agents(){
	return d_WorkplaceMemberships_wpmembershipdefault;
}

xmachine_memory_WorkplaceMembership_list* get_host_WorkplaceMembership_wpmembershipdefault_agents(){
	return h_WorkplaceMemberships_wpmembershipdefault;
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

/** unsigned int get_Person_default_variable_householdtime(unsigned int index)
 * Gets the value of the householdtime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdtime
 */
__host__ unsigned int get_Person_default_variable_householdtime(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_householdtime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->householdtime,
                    d_Persons_default->householdtime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_householdtime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->householdtime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access householdtime for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_churchtime(unsigned int index)
 * Gets the value of the churchtime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchtime
 */
__host__ unsigned int get_Person_default_variable_churchtime(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_churchtime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->churchtime,
                    d_Persons_default->churchtime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_churchtime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->churchtime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchtime for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_transporttime(unsigned int index)
 * Gets the value of the transporttime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transporttime
 */
__host__ unsigned int get_Person_default_variable_transporttime(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transporttime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transporttime,
                    d_Persons_default->transporttime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transporttime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transporttime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transporttime for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_clinictime(unsigned int index)
 * Gets the value of the clinictime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable clinictime
 */
__host__ unsigned int get_Person_default_variable_clinictime(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_clinictime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->clinictime,
                    d_Persons_default->clinictime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_clinictime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->clinictime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access clinictime for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_workplacetime(unsigned int index)
 * Gets the value of the workplacetime variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplacetime
 */
__host__ unsigned int get_Person_default_variable_workplacetime(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_workplacetime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->workplacetime,
                    d_Persons_default->workplacetime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_workplacetime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->workplacetime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access workplacetime for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_default_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ unsigned int get_Person_default_variable_transportdur(unsigned int index){
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
                    count * sizeof(unsigned int),
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

/** int get_Person_default_variable_transportday1(unsigned int index)
 * Gets the value of the transportday1 variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday1
 */
__host__ int get_Person_default_variable_transportday1(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transportday1_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transportday1,
                    d_Persons_default->transportday1,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transportday1_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transportday1[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportday1 for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_transportday2(unsigned int index)
 * Gets the value of the transportday2 variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday2
 */
__host__ int get_Person_default_variable_transportday2(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transportday2_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transportday2,
                    d_Persons_default->transportday2,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transportday2_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transportday2[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportday2 for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** int get_Person_default_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ int get_Person_default_variable_church(unsigned int index){
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
                    count * sizeof(int),
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

/** int get_Person_default_variable_transport(unsigned int index)
 * Gets the value of the transport variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport
 */
__host__ int get_Person_default_variable_transport(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_transport_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->transport,
                    d_Persons_default->transport,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_transport_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->transport[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transport for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_workplace(unsigned int index)
 * Gets the value of the workplace variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace
 */
__host__ int get_Person_default_variable_workplace(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_workplace_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->workplace,
                    d_Persons_default->workplace,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_workplace_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->workplace[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access workplace for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_default_variable_location(unsigned int index)
 * Gets the value of the location variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location
 */
__host__ unsigned int get_Person_default_variable_location(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_location_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->location,
                    d_Persons_default->location,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_location_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->location[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access location for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_locationid(unsigned int index)
 * Gets the value of the locationid variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable locationid
 */
__host__ unsigned int get_Person_default_variable_locationid(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_locationid_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->locationid,
                    d_Persons_default->locationid,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_locationid_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->locationid[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access locationid for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_hiv(unsigned int index)
 * Gets the value of the hiv variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hiv
 */
__host__ unsigned int get_Person_default_variable_hiv(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_hiv_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->hiv,
                    d_Persons_default->hiv,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_hiv_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->hiv[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hiv for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_art(unsigned int index)
 * Gets the value of the art variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable art
 */
__host__ unsigned int get_Person_default_variable_art(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_art_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->art,
                    d_Persons_default->art,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_art_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->art[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access art for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_activetb(unsigned int index)
 * Gets the value of the activetb variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable activetb
 */
__host__ unsigned int get_Person_default_variable_activetb(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_activetb_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->activetb,
                    d_Persons_default->activetb,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_activetb_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->activetb[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access activetb for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_artday(unsigned int index)
 * Gets the value of the artday variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable artday
 */
__host__ unsigned int get_Person_default_variable_artday(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_artday_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->artday,
                    d_Persons_default->artday,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_artday_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->artday[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access artday for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_default_variable_p(unsigned int index)
 * Gets the value of the p variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable p
 */
__host__ float get_Person_default_variable_p(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_p_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->p,
                    d_Persons_default->p,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_p_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->p[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access p for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_default_variable_q(unsigned int index)
 * Gets the value of the q variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable q
 */
__host__ float get_Person_default_variable_q(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_q_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->q,
                    d_Persons_default->q,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_q_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->q[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access q for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_infections(unsigned int index)
 * Gets the value of the infections variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable infections
 */
__host__ unsigned int get_Person_default_variable_infections(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_infections_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->infections,
                    d_Persons_default->infections,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_infections_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->infections[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access infections for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_lastinfected(unsigned int index)
 * Gets the value of the lastinfected variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfected
 */
__host__ int get_Person_default_variable_lastinfected(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_lastinfected_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->lastinfected,
                    d_Persons_default->lastinfected,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_lastinfected_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->lastinfected[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lastinfected for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_default_variable_lastinfectedid(unsigned int index)
 * Gets the value of the lastinfectedid variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedid
 */
__host__ int get_Person_default_variable_lastinfectedid(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_lastinfectedid_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->lastinfectedid,
                    d_Persons_default->lastinfectedid,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_lastinfectedid_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->lastinfectedid[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lastinfectedid for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_default_variable_time_step(unsigned int index)
 * Gets the value of the time_step variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable time_step
 */
__host__ float get_Person_default_variable_time_step(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_time_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->time_step,
                    d_Persons_default->time_step,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_time_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->time_step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access time_step for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_default_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Person_default_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->lambda,
                    d_Persons_default->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_default_variable_timevisiting(unsigned int index)
 * Gets the value of the timevisiting variable of an Person agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timevisiting
 */
__host__ unsigned int get_Person_default_variable_timevisiting(unsigned int index){
    unsigned int count = get_agent_Person_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_default_variable_timevisiting_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_default->timevisiting,
                    d_Persons_default->timevisiting,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_default_variable_timevisiting_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_default->timevisiting[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access timevisiting for the %u th member of Person_default. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_s2_variable_householdtime(unsigned int index)
 * Gets the value of the householdtime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable householdtime
 */
__host__ unsigned int get_Person_s2_variable_householdtime(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_householdtime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->householdtime,
                    d_Persons_s2->householdtime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_householdtime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->householdtime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access householdtime for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_churchtime(unsigned int index)
 * Gets the value of the churchtime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable churchtime
 */
__host__ unsigned int get_Person_s2_variable_churchtime(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_churchtime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->churchtime,
                    d_Persons_s2->churchtime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_churchtime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->churchtime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access churchtime for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_transporttime(unsigned int index)
 * Gets the value of the transporttime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transporttime
 */
__host__ unsigned int get_Person_s2_variable_transporttime(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transporttime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transporttime,
                    d_Persons_s2->transporttime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transporttime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transporttime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transporttime for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_clinictime(unsigned int index)
 * Gets the value of the clinictime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable clinictime
 */
__host__ unsigned int get_Person_s2_variable_clinictime(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_clinictime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->clinictime,
                    d_Persons_s2->clinictime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_clinictime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->clinictime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access clinictime for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_workplacetime(unsigned int index)
 * Gets the value of the workplacetime variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplacetime
 */
__host__ unsigned int get_Person_s2_variable_workplacetime(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_workplacetime_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->workplacetime,
                    d_Persons_s2->workplacetime,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_workplacetime_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->workplacetime[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access workplacetime for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_s2_variable_transportdur(unsigned int index)
 * Gets the value of the transportdur variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportdur
 */
__host__ unsigned int get_Person_s2_variable_transportdur(unsigned int index){
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
                    count * sizeof(unsigned int),
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

/** int get_Person_s2_variable_transportday1(unsigned int index)
 * Gets the value of the transportday1 variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday1
 */
__host__ int get_Person_s2_variable_transportday1(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transportday1_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transportday1,
                    d_Persons_s2->transportday1,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transportday1_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transportday1[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportday1 for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_transportday2(unsigned int index)
 * Gets the value of the transportday2 variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transportday2
 */
__host__ int get_Person_s2_variable_transportday2(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transportday2_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transportday2,
                    d_Persons_s2->transportday2,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transportday2_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transportday2[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transportday2 for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** int get_Person_s2_variable_church(unsigned int index)
 * Gets the value of the church variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable church
 */
__host__ int get_Person_s2_variable_church(unsigned int index){
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
                    count * sizeof(int),
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

/** int get_Person_s2_variable_transport(unsigned int index)
 * Gets the value of the transport variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport
 */
__host__ int get_Person_s2_variable_transport(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_transport_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->transport,
                    d_Persons_s2->transport,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_transport_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->transport[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transport for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_workplace(unsigned int index)
 * Gets the value of the workplace variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace
 */
__host__ int get_Person_s2_variable_workplace(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_workplace_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->workplace,
                    d_Persons_s2->workplace,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_workplace_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->workplace[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access workplace for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_Person_s2_variable_location(unsigned int index)
 * Gets the value of the location variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location
 */
__host__ unsigned int get_Person_s2_variable_location(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_location_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->location,
                    d_Persons_s2->location,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_location_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->location[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access location for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_locationid(unsigned int index)
 * Gets the value of the locationid variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable locationid
 */
__host__ unsigned int get_Person_s2_variable_locationid(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_locationid_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->locationid,
                    d_Persons_s2->locationid,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_locationid_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->locationid[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access locationid for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_hiv(unsigned int index)
 * Gets the value of the hiv variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hiv
 */
__host__ unsigned int get_Person_s2_variable_hiv(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_hiv_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->hiv,
                    d_Persons_s2->hiv,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_hiv_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->hiv[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hiv for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_art(unsigned int index)
 * Gets the value of the art variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable art
 */
__host__ unsigned int get_Person_s2_variable_art(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_art_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->art,
                    d_Persons_s2->art,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_art_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->art[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access art for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_activetb(unsigned int index)
 * Gets the value of the activetb variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable activetb
 */
__host__ unsigned int get_Person_s2_variable_activetb(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_activetb_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->activetb,
                    d_Persons_s2->activetb,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_activetb_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->activetb[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access activetb for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_artday(unsigned int index)
 * Gets the value of the artday variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable artday
 */
__host__ unsigned int get_Person_s2_variable_artday(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_artday_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->artday,
                    d_Persons_s2->artday,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_artday_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->artday[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access artday for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_s2_variable_p(unsigned int index)
 * Gets the value of the p variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable p
 */
__host__ float get_Person_s2_variable_p(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_p_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->p,
                    d_Persons_s2->p,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_p_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->p[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access p for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_s2_variable_q(unsigned int index)
 * Gets the value of the q variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable q
 */
__host__ float get_Person_s2_variable_q(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_q_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->q,
                    d_Persons_s2->q,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_q_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->q[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access q for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_infections(unsigned int index)
 * Gets the value of the infections variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable infections
 */
__host__ unsigned int get_Person_s2_variable_infections(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_infections_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->infections,
                    d_Persons_s2->infections,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_infections_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->infections[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access infections for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_lastinfected(unsigned int index)
 * Gets the value of the lastinfected variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfected
 */
__host__ int get_Person_s2_variable_lastinfected(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_lastinfected_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->lastinfected,
                    d_Persons_s2->lastinfected,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_lastinfected_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->lastinfected[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lastinfected for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Person_s2_variable_lastinfectedid(unsigned int index)
 * Gets the value of the lastinfectedid variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lastinfectedid
 */
__host__ int get_Person_s2_variable_lastinfectedid(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_lastinfectedid_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->lastinfectedid,
                    d_Persons_s2->lastinfectedid,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_lastinfectedid_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->lastinfectedid[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lastinfectedid for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_s2_variable_time_step(unsigned int index)
 * Gets the value of the time_step variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable time_step
 */
__host__ float get_Person_s2_variable_time_step(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_time_step_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->time_step,
                    d_Persons_s2->time_step,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_time_step_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->time_step[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access time_step for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Person_s2_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Person_s2_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->lambda,
                    d_Persons_s2->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Person_s2_variable_timevisiting(unsigned int index)
 * Gets the value of the timevisiting variable of an Person agent in the s2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable timevisiting
 */
__host__ unsigned int get_Person_s2_variable_timevisiting(unsigned int index){
    unsigned int count = get_agent_Person_s2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Persons_s2_variable_timevisiting_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Persons_s2->timevisiting,
                    d_Persons_s2->timevisiting,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Persons_s2_variable_timevisiting_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Persons_s2->timevisiting[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access timevisiting for the %u th member of Person_s2. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_TBAssignment_tbdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an TBAssignment agent in the tbdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_TBAssignment_tbdefault_variable_id(unsigned int index){
    unsigned int count = get_agent_TBAssignment_tbdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TBAssignments_tbdefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_TBAssignments_tbdefault->id,
                    d_TBAssignments_tbdefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TBAssignments_tbdefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TBAssignments_tbdefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of TBAssignment_tbdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** float get_Household_hhdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Household_hhdefault_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->lambda,
                    d_Households_hhdefault->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Household_hhdefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Household agent in the hhdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Household_hhdefault_variable_active(unsigned int index){
    unsigned int count = get_agent_Household_hhdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Households_hhdefault_variable_active_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Households_hhdefault->active,
                    d_Households_hhdefault->active,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Households_hhdefault_variable_active_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Households_hhdefault->active[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access active for the %u th member of Household_hhdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_size(unsigned int index)
 * Gets the value of the household_size variable of an HouseholdMembership agent in the hhmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable household_size
 */
__host__ unsigned int get_HouseholdMembership_hhmembershipdefault_variable_household_size(unsigned int index){
    unsigned int count = get_agent_HouseholdMembership_hhmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_HouseholdMemberships_hhmembershipdefault->household_size,
                    d_HouseholdMemberships_hhmembershipdefault->household_size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_HouseholdMemberships_hhmembershipdefault->household_size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access household_size for the %u th member of HouseholdMembership_hhmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** float get_Church_chudefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Church_chudefault_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->lambda,
                    d_Churchs_chudefault->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Church_chudefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Church agent in the chudefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Church_chudefault_variable_active(unsigned int index){
    unsigned int count = get_agent_Church_chudefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Churchs_chudefault_variable_active_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Churchs_chudefault->active,
                    d_Churchs_chudefault->active,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Churchs_chudefault_variable_active_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Churchs_chudefault->active[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access active for the %u th member of Church_chudefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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

/** float get_Transport_trdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Transport_trdefault_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Transport_trdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Transports_trdefault_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Transports_trdefault->lambda,
                    d_Transports_trdefault->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Transports_trdefault_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Transports_trdefault->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Transport_trdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Transport_trdefault_variable_active(unsigned int index)
 * Gets the value of the active variable of an Transport agent in the trdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable active
 */
__host__ unsigned int get_Transport_trdefault_variable_active(unsigned int index){
    unsigned int count = get_agent_Transport_trdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Transports_trdefault_variable_active_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Transports_trdefault->active,
                    d_Transports_trdefault->active,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Transports_trdefault_variable_active_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Transports_trdefault->active[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access active for the %u th member of Transport_trdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_TransportMembership_trmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ int get_TransportMembership_trmembershipdefault_variable_person_id(unsigned int index){
    unsigned int count = get_agent_TransportMembership_trmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_TransportMemberships_trmembershipdefault->person_id,
                    d_TransportMemberships_trmembershipdefault->person_id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TransportMemberships_trmembershipdefault->person_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access person_id for the %u th member of TransportMembership_trmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_TransportMembership_trmembershipdefault_variable_transport_id(unsigned int index)
 * Gets the value of the transport_id variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable transport_id
 */
__host__ unsigned int get_TransportMembership_trmembershipdefault_variable_transport_id(unsigned int index){
    unsigned int count = get_agent_TransportMembership_trmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_TransportMemberships_trmembershipdefault->transport_id,
                    d_TransportMemberships_trmembershipdefault->transport_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TransportMemberships_trmembershipdefault->transport_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access transport_id for the %u th member of TransportMembership_trmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_TransportMembership_trmembershipdefault_variable_duration(unsigned int index)
 * Gets the value of the duration variable of an TransportMembership agent in the trmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable duration
 */
__host__ unsigned int get_TransportMembership_trmembershipdefault_variable_duration(unsigned int index){
    unsigned int count = get_agent_TransportMembership_trmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_TransportMemberships_trmembershipdefault->duration,
                    d_TransportMemberships_trmembershipdefault->duration,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_TransportMemberships_trmembershipdefault->duration[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access duration for the %u th member of TransportMembership_trmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Clinic_cldefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Clinic agent in the cldefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Clinic_cldefault_variable_id(unsigned int index){
    unsigned int count = get_agent_Clinic_cldefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Clinics_cldefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Clinics_cldefault->id,
                    d_Clinics_cldefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Clinics_cldefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Clinics_cldefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Clinic_cldefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Clinic_cldefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Clinic agent in the cldefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Clinic_cldefault_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Clinic_cldefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Clinics_cldefault_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Clinics_cldefault->lambda,
                    d_Clinics_cldefault->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Clinics_cldefault_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Clinics_cldefault->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Clinic_cldefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Workplace_wpdefault_variable_id(unsigned int index)
 * Gets the value of the id variable of an Workplace agent in the wpdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Workplace_wpdefault_variable_id(unsigned int index){
    unsigned int count = get_agent_Workplace_wpdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Workplaces_wpdefault_variable_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Workplaces_wpdefault->id,
                    d_Workplaces_wpdefault->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Workplaces_wpdefault_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Workplaces_wpdefault->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Workplace_wpdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Workplace_wpdefault_variable_lambda(unsigned int index)
 * Gets the value of the lambda variable of an Workplace agent in the wpdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lambda
 */
__host__ float get_Workplace_wpdefault_variable_lambda(unsigned int index){
    unsigned int count = get_agent_Workplace_wpdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Workplaces_wpdefault_variable_lambda_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_Workplaces_wpdefault->lambda,
                    d_Workplaces_wpdefault->lambda,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Workplaces_wpdefault_variable_lambda_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Workplaces_wpdefault->lambda[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lambda for the %u th member of Workplace_wpdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_person_id(unsigned int index)
 * Gets the value of the person_id variable of an WorkplaceMembership agent in the wpmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable person_id
 */
__host__ unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_person_id(unsigned int index){
    unsigned int count = get_agent_WorkplaceMembership_wpmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_WorkplaceMemberships_wpmembershipdefault->person_id,
                    d_WorkplaceMemberships_wpmembershipdefault->person_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_WorkplaceMemberships_wpmembershipdefault->person_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access person_id for the %u th member of WorkplaceMembership_wpmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_workplace_id(unsigned int index)
 * Gets the value of the workplace_id variable of an WorkplaceMembership agent in the wpmembershipdefault state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable workplace_id
 */
__host__ unsigned int get_WorkplaceMembership_wpmembershipdefault_variable_workplace_id(unsigned int index){
    unsigned int count = get_agent_WorkplaceMembership_wpmembershipdefault_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration != currentIteration){
            
            gpuErrchk(
                cudaMemcpy(
                    h_WorkplaceMemberships_wpmembershipdefault->workplace_id,
                    d_WorkplaceMemberships_wpmembershipdefault->workplace_id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_WorkplaceMemberships_wpmembershipdefault->workplace_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access workplace_id for the %u th member of WorkplaceMembership_wpmembershipdefault. count is %u at iteration %u\n", index, count, currentIteration); //@todo
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
 
		gpuErrchk(cudaMemcpy(d_dst->householdtime, &h_agent->householdtime, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchtime, &h_agent->churchtime, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transporttime, &h_agent->transporttime, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->clinictime, &h_agent->clinictime, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplacetime, &h_agent->workplacetime, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, &h_agent->age, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, &h_agent->gender, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, &h_agent->householdsize, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, &h_agent->churchfreq, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, &h_agent->churchdur, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportdur, &h_agent->transportdur, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportday1, &h_agent->transportday1, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportday2, &h_agent->transportday2, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household, &h_agent->household, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->church, &h_agent->church, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transport, &h_agent->transport, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplace, &h_agent->workplace, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->busy, &h_agent->busy, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->startstep, &h_agent->startstep, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->location, &h_agent->location, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->locationid, &h_agent->locationid, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->hiv, &h_agent->hiv, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->art, &h_agent->art, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->activetb, &h_agent->activetb, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->artday, &h_agent->artday, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->p, &h_agent->p, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->q, &h_agent->q, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->infections, &h_agent->infections, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lastinfected, &h_agent->lastinfected, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lastinfectedid, &h_agent->lastinfectedid, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->time_step, &h_agent->time_step, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->timevisiting, &h_agent->timevisiting, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->householdtime, h_src->householdtime, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchtime, h_src->churchtime, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transporttime, h_src->transporttime, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->clinictime, h_src->clinictime, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplacetime, h_src->workplacetime, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->age, h_src->age, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->gender, h_src->gender, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->householdsize, h_src->householdsize, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchfreq, h_src->churchfreq, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->churchdur, h_src->churchdur, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportdur, h_src->transportdur, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportday1, h_src->transportday1, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transportday2, h_src->transportday2, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->household, h_src->household, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->church, h_src->church, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transport, h_src->transport, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplace, h_src->workplace, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->busy, h_src->busy, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->startstep, h_src->startstep, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->location, h_src->location, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->locationid, h_src->locationid, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->hiv, h_src->hiv, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->art, h_src->art, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->activetb, h_src->activetb, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->artday, h_src->artday, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->p, h_src->p, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->q, h_src->q, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->infections, h_src->infections, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lastinfected, h_src->lastinfected, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lastinfectedid, h_src->lastinfectedid, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->time_step, h_src->time_step, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->timevisiting, h_src->timevisiting, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_TBAssignment_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_TBAssignment_hostToDevice(xmachine_memory_TBAssignment_list * d_dst, xmachine_memory_TBAssignment * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_TBAssignment_hostToDevice(xmachine_memory_TBAssignment_list * d_dst, xmachine_memory_TBAssignment_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Household_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Household_hostToDevice(xmachine_memory_Household_list * d_dst, xmachine_memory_Household * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, &h_agent->active, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, h_src->active, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->household_size, &h_agent->household_size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
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
 
		gpuErrchk(cudaMemcpy(d_dst->household_size, h_src->household_size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
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
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, &h_agent->active, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, h_src->active, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, &h_agent->active, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->active, h_src->active, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_TransportMembership_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_TransportMembership_hostToDevice(xmachine_memory_TransportMembership_list * d_dst, xmachine_memory_TransportMembership * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->person_id, &h_agent->person_id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transport_id, &h_agent->transport_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
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
void copy_partial_xmachine_memory_TransportMembership_hostToDevice(xmachine_memory_TransportMembership_list * d_dst, xmachine_memory_TransportMembership_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->person_id, h_src->person_id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->transport_id, h_src->transport_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->duration, h_src->duration, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Clinic_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Clinic_hostToDevice(xmachine_memory_Clinic_list * d_dst, xmachine_memory_Clinic * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_Clinic_hostToDevice(xmachine_memory_Clinic_list * d_dst, xmachine_memory_Clinic_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_Workplace_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Workplace_hostToDevice(xmachine_memory_Workplace_list * d_dst, xmachine_memory_Workplace * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, &h_agent->lambda, sizeof(float), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_Workplace_hostToDevice(xmachine_memory_Workplace_list * d_dst, xmachine_memory_Workplace_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lambda, h_src->lambda, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_WorkplaceMembership_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_WorkplaceMembership_hostToDevice(xmachine_memory_WorkplaceMembership_list * d_dst, xmachine_memory_WorkplaceMembership * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->person_id, &h_agent->person_id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplace_id, &h_agent->workplace_id, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_WorkplaceMembership_hostToDevice(xmachine_memory_WorkplaceMembership_list * d_dst, xmachine_memory_WorkplaceMembership_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->person_id, h_src->person_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->workplace_id, h_src->workplace_id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
			 
			dst->householdtime[i] = src[i]->householdtime;
			 
			dst->churchtime[i] = src[i]->churchtime;
			 
			dst->transporttime[i] = src[i]->transporttime;
			 
			dst->clinictime[i] = src[i]->clinictime;
			 
			dst->workplacetime[i] = src[i]->workplacetime;
			 
			dst->age[i] = src[i]->age;
			 
			dst->gender[i] = src[i]->gender;
			 
			dst->householdsize[i] = src[i]->householdsize;
			 
			dst->churchfreq[i] = src[i]->churchfreq;
			 
			dst->churchdur[i] = src[i]->churchdur;
			 
			dst->transportdur[i] = src[i]->transportdur;
			 
			dst->transportday1[i] = src[i]->transportday1;
			 
			dst->transportday2[i] = src[i]->transportday2;
			 
			dst->household[i] = src[i]->household;
			 
			dst->church[i] = src[i]->church;
			 
			dst->transport[i] = src[i]->transport;
			 
			dst->workplace[i] = src[i]->workplace;
			 
			dst->busy[i] = src[i]->busy;
			 
			dst->startstep[i] = src[i]->startstep;
			 
			dst->location[i] = src[i]->location;
			 
			dst->locationid[i] = src[i]->locationid;
			 
			dst->hiv[i] = src[i]->hiv;
			 
			dst->art[i] = src[i]->art;
			 
			dst->activetb[i] = src[i]->activetb;
			 
			dst->artday[i] = src[i]->artday;
			 
			dst->p[i] = src[i]->p;
			 
			dst->q[i] = src[i]->q;
			 
			dst->infections[i] = src[i]->infections;
			 
			dst->lastinfected[i] = src[i]->lastinfected;
			 
			dst->lastinfectedid[i] = src[i]->lastinfectedid;
			 
			dst->time_step[i] = src[i]->time_step;
			 
			dst->lambda[i] = src[i]->lambda;
			 
			dst->timevisiting[i] = src[i]->timevisiting;
			
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
    h_Persons_default_variable_householdtime_data_iteration = 0;
    h_Persons_default_variable_churchtime_data_iteration = 0;
    h_Persons_default_variable_transporttime_data_iteration = 0;
    h_Persons_default_variable_clinictime_data_iteration = 0;
    h_Persons_default_variable_workplacetime_data_iteration = 0;
    h_Persons_default_variable_age_data_iteration = 0;
    h_Persons_default_variable_gender_data_iteration = 0;
    h_Persons_default_variable_householdsize_data_iteration = 0;
    h_Persons_default_variable_churchfreq_data_iteration = 0;
    h_Persons_default_variable_churchdur_data_iteration = 0;
    h_Persons_default_variable_transportdur_data_iteration = 0;
    h_Persons_default_variable_transportday1_data_iteration = 0;
    h_Persons_default_variable_transportday2_data_iteration = 0;
    h_Persons_default_variable_household_data_iteration = 0;
    h_Persons_default_variable_church_data_iteration = 0;
    h_Persons_default_variable_transport_data_iteration = 0;
    h_Persons_default_variable_workplace_data_iteration = 0;
    h_Persons_default_variable_busy_data_iteration = 0;
    h_Persons_default_variable_startstep_data_iteration = 0;
    h_Persons_default_variable_location_data_iteration = 0;
    h_Persons_default_variable_locationid_data_iteration = 0;
    h_Persons_default_variable_hiv_data_iteration = 0;
    h_Persons_default_variable_art_data_iteration = 0;
    h_Persons_default_variable_activetb_data_iteration = 0;
    h_Persons_default_variable_artday_data_iteration = 0;
    h_Persons_default_variable_p_data_iteration = 0;
    h_Persons_default_variable_q_data_iteration = 0;
    h_Persons_default_variable_infections_data_iteration = 0;
    h_Persons_default_variable_lastinfected_data_iteration = 0;
    h_Persons_default_variable_lastinfectedid_data_iteration = 0;
    h_Persons_default_variable_time_step_data_iteration = 0;
    h_Persons_default_variable_lambda_data_iteration = 0;
    h_Persons_default_variable_timevisiting_data_iteration = 0;
    

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
        h_Persons_default_variable_householdtime_data_iteration = 0;
        h_Persons_default_variable_churchtime_data_iteration = 0;
        h_Persons_default_variable_transporttime_data_iteration = 0;
        h_Persons_default_variable_clinictime_data_iteration = 0;
        h_Persons_default_variable_workplacetime_data_iteration = 0;
        h_Persons_default_variable_age_data_iteration = 0;
        h_Persons_default_variable_gender_data_iteration = 0;
        h_Persons_default_variable_householdsize_data_iteration = 0;
        h_Persons_default_variable_churchfreq_data_iteration = 0;
        h_Persons_default_variable_churchdur_data_iteration = 0;
        h_Persons_default_variable_transportdur_data_iteration = 0;
        h_Persons_default_variable_transportday1_data_iteration = 0;
        h_Persons_default_variable_transportday2_data_iteration = 0;
        h_Persons_default_variable_household_data_iteration = 0;
        h_Persons_default_variable_church_data_iteration = 0;
        h_Persons_default_variable_transport_data_iteration = 0;
        h_Persons_default_variable_workplace_data_iteration = 0;
        h_Persons_default_variable_busy_data_iteration = 0;
        h_Persons_default_variable_startstep_data_iteration = 0;
        h_Persons_default_variable_location_data_iteration = 0;
        h_Persons_default_variable_locationid_data_iteration = 0;
        h_Persons_default_variable_hiv_data_iteration = 0;
        h_Persons_default_variable_art_data_iteration = 0;
        h_Persons_default_variable_activetb_data_iteration = 0;
        h_Persons_default_variable_artday_data_iteration = 0;
        h_Persons_default_variable_p_data_iteration = 0;
        h_Persons_default_variable_q_data_iteration = 0;
        h_Persons_default_variable_infections_data_iteration = 0;
        h_Persons_default_variable_lastinfected_data_iteration = 0;
        h_Persons_default_variable_lastinfectedid_data_iteration = 0;
        h_Persons_default_variable_time_step_data_iteration = 0;
        h_Persons_default_variable_lambda_data_iteration = 0;
        h_Persons_default_variable_timevisiting_data_iteration = 0;
        

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
    h_Persons_s2_variable_householdtime_data_iteration = 0;
    h_Persons_s2_variable_churchtime_data_iteration = 0;
    h_Persons_s2_variable_transporttime_data_iteration = 0;
    h_Persons_s2_variable_clinictime_data_iteration = 0;
    h_Persons_s2_variable_workplacetime_data_iteration = 0;
    h_Persons_s2_variable_age_data_iteration = 0;
    h_Persons_s2_variable_gender_data_iteration = 0;
    h_Persons_s2_variable_householdsize_data_iteration = 0;
    h_Persons_s2_variable_churchfreq_data_iteration = 0;
    h_Persons_s2_variable_churchdur_data_iteration = 0;
    h_Persons_s2_variable_transportdur_data_iteration = 0;
    h_Persons_s2_variable_transportday1_data_iteration = 0;
    h_Persons_s2_variable_transportday2_data_iteration = 0;
    h_Persons_s2_variable_household_data_iteration = 0;
    h_Persons_s2_variable_church_data_iteration = 0;
    h_Persons_s2_variable_transport_data_iteration = 0;
    h_Persons_s2_variable_workplace_data_iteration = 0;
    h_Persons_s2_variable_busy_data_iteration = 0;
    h_Persons_s2_variable_startstep_data_iteration = 0;
    h_Persons_s2_variable_location_data_iteration = 0;
    h_Persons_s2_variable_locationid_data_iteration = 0;
    h_Persons_s2_variable_hiv_data_iteration = 0;
    h_Persons_s2_variable_art_data_iteration = 0;
    h_Persons_s2_variable_activetb_data_iteration = 0;
    h_Persons_s2_variable_artday_data_iteration = 0;
    h_Persons_s2_variable_p_data_iteration = 0;
    h_Persons_s2_variable_q_data_iteration = 0;
    h_Persons_s2_variable_infections_data_iteration = 0;
    h_Persons_s2_variable_lastinfected_data_iteration = 0;
    h_Persons_s2_variable_lastinfectedid_data_iteration = 0;
    h_Persons_s2_variable_time_step_data_iteration = 0;
    h_Persons_s2_variable_lambda_data_iteration = 0;
    h_Persons_s2_variable_timevisiting_data_iteration = 0;
    

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
        h_Persons_s2_variable_householdtime_data_iteration = 0;
        h_Persons_s2_variable_churchtime_data_iteration = 0;
        h_Persons_s2_variable_transporttime_data_iteration = 0;
        h_Persons_s2_variable_clinictime_data_iteration = 0;
        h_Persons_s2_variable_workplacetime_data_iteration = 0;
        h_Persons_s2_variable_age_data_iteration = 0;
        h_Persons_s2_variable_gender_data_iteration = 0;
        h_Persons_s2_variable_householdsize_data_iteration = 0;
        h_Persons_s2_variable_churchfreq_data_iteration = 0;
        h_Persons_s2_variable_churchdur_data_iteration = 0;
        h_Persons_s2_variable_transportdur_data_iteration = 0;
        h_Persons_s2_variable_transportday1_data_iteration = 0;
        h_Persons_s2_variable_transportday2_data_iteration = 0;
        h_Persons_s2_variable_household_data_iteration = 0;
        h_Persons_s2_variable_church_data_iteration = 0;
        h_Persons_s2_variable_transport_data_iteration = 0;
        h_Persons_s2_variable_workplace_data_iteration = 0;
        h_Persons_s2_variable_busy_data_iteration = 0;
        h_Persons_s2_variable_startstep_data_iteration = 0;
        h_Persons_s2_variable_location_data_iteration = 0;
        h_Persons_s2_variable_locationid_data_iteration = 0;
        h_Persons_s2_variable_hiv_data_iteration = 0;
        h_Persons_s2_variable_art_data_iteration = 0;
        h_Persons_s2_variable_activetb_data_iteration = 0;
        h_Persons_s2_variable_artday_data_iteration = 0;
        h_Persons_s2_variable_p_data_iteration = 0;
        h_Persons_s2_variable_q_data_iteration = 0;
        h_Persons_s2_variable_infections_data_iteration = 0;
        h_Persons_s2_variable_lastinfected_data_iteration = 0;
        h_Persons_s2_variable_lastinfectedid_data_iteration = 0;
        h_Persons_s2_variable_time_step_data_iteration = 0;
        h_Persons_s2_variable_lambda_data_iteration = 0;
        h_Persons_s2_variable_timevisiting_data_iteration = 0;
        

	}
}

xmachine_memory_TBAssignment* h_allocate_agent_TBAssignment(){
	xmachine_memory_TBAssignment* agent = (xmachine_memory_TBAssignment*)malloc(sizeof(xmachine_memory_TBAssignment));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_TBAssignment));

	return agent;
}
void h_free_agent_TBAssignment(xmachine_memory_TBAssignment** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_TBAssignment** h_allocate_agent_TBAssignment_array(unsigned int count){
	xmachine_memory_TBAssignment ** agents = (xmachine_memory_TBAssignment**)malloc(count * sizeof(xmachine_memory_TBAssignment*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_TBAssignment();
	}
	return agents;
}
void h_free_agent_TBAssignment_array(xmachine_memory_TBAssignment*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_TBAssignment(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_TBAssignment_AoS_to_SoA(xmachine_memory_TBAssignment_list * dst, xmachine_memory_TBAssignment** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			
		}
	}
}


void h_add_agent_TBAssignment_tbdefault(xmachine_memory_TBAssignment* agent){
	if (h_xmachine_memory_TBAssignment_count + 1 > xmachine_memory_TBAssignment_MAX){
		printf("Error: Buffer size of TBAssignment agents in state tbdefault will be exceeded by h_add_agent_TBAssignment_tbdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_TBAssignment_hostToDevice(d_TBAssignments_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TBAssignment_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_TBAssignment_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TBAssignments_tbdefault, d_TBAssignments_new, h_xmachine_memory_TBAssignment_tbdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_TBAssignment_tbdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TBAssignment_tbdefault_count, &h_xmachine_memory_TBAssignment_tbdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_TBAssignments_tbdefault_variable_id_data_iteration = 0;
    

}
void h_add_agents_TBAssignment_tbdefault(xmachine_memory_TBAssignment** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_TBAssignment_count + count > xmachine_memory_TBAssignment_MAX){
			printf("Error: Buffer size of TBAssignment agents in state tbdefault will be exceeded by h_add_agents_TBAssignment_tbdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_TBAssignment_AoS_to_SoA(h_TBAssignments_tbdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_TBAssignment_hostToDevice(d_TBAssignments_new, h_TBAssignments_tbdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TBAssignment_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_TBAssignment_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TBAssignments_tbdefault, d_TBAssignments_new, h_xmachine_memory_TBAssignment_tbdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_TBAssignment_tbdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TBAssignment_tbdefault_count, &h_xmachine_memory_TBAssignment_tbdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_TBAssignments_tbdefault_variable_id_data_iteration = 0;
        

	}
}

xmachine_memory_Household* h_allocate_agent_Household(){
	xmachine_memory_Household* agent = (xmachine_memory_Household*)malloc(sizeof(xmachine_memory_Household));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Household));

	return agent;
}
void h_free_agent_Household(xmachine_memory_Household** agent){
 
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
			 
			dst->lambda[i] = src[i]->lambda;
			 
			dst->active[i] = src[i]->active;
			
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
    h_Households_hhdefault_variable_lambda_data_iteration = 0;
    h_Households_hhdefault_variable_active_data_iteration = 0;
    

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
        h_Households_hhdefault_variable_lambda_data_iteration = 0;
        h_Households_hhdefault_variable_active_data_iteration = 0;
        

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
			 
			dst->household_size[i] = src[i]->household_size;
			 
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
    h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration = 0;
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
        h_HouseholdMemberships_hhmembershipdefault_variable_household_size_data_iteration = 0;
        h_HouseholdMemberships_hhmembershipdefault_variable_churchgoing_data_iteration = 0;
        h_HouseholdMemberships_hhmembershipdefault_variable_churchfreq_data_iteration = 0;
        

	}
}

xmachine_memory_Church* h_allocate_agent_Church(){
	xmachine_memory_Church* agent = (xmachine_memory_Church*)malloc(sizeof(xmachine_memory_Church));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Church));

	return agent;
}
void h_free_agent_Church(xmachine_memory_Church** agent){
 
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
			 
			dst->size[i] = src[i]->size;
			 
			dst->lambda[i] = src[i]->lambda;
			 
			dst->active[i] = src[i]->active;
			
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
    h_Churchs_chudefault_variable_size_data_iteration = 0;
    h_Churchs_chudefault_variable_lambda_data_iteration = 0;
    h_Churchs_chudefault_variable_active_data_iteration = 0;
    

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
        h_Churchs_chudefault_variable_size_data_iteration = 0;
        h_Churchs_chudefault_variable_lambda_data_iteration = 0;
        h_Churchs_chudefault_variable_active_data_iteration = 0;
        

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
			 
			dst->lambda[i] = src[i]->lambda;
			 
			dst->active[i] = src[i]->active;
			
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
    h_Transports_trdefault_variable_lambda_data_iteration = 0;
    h_Transports_trdefault_variable_active_data_iteration = 0;
    

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
        h_Transports_trdefault_variable_lambda_data_iteration = 0;
        h_Transports_trdefault_variable_active_data_iteration = 0;
        

	}
}

xmachine_memory_TransportMembership* h_allocate_agent_TransportMembership(){
	xmachine_memory_TransportMembership* agent = (xmachine_memory_TransportMembership*)malloc(sizeof(xmachine_memory_TransportMembership));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_TransportMembership));

	return agent;
}
void h_free_agent_TransportMembership(xmachine_memory_TransportMembership** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_TransportMembership** h_allocate_agent_TransportMembership_array(unsigned int count){
	xmachine_memory_TransportMembership ** agents = (xmachine_memory_TransportMembership**)malloc(count * sizeof(xmachine_memory_TransportMembership*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_TransportMembership();
	}
	return agents;
}
void h_free_agent_TransportMembership_array(xmachine_memory_TransportMembership*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_TransportMembership(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_TransportMembership_AoS_to_SoA(xmachine_memory_TransportMembership_list * dst, xmachine_memory_TransportMembership** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->person_id[i] = src[i]->person_id;
			 
			dst->transport_id[i] = src[i]->transport_id;
			 
			dst->duration[i] = src[i]->duration;
			
		}
	}
}


void h_add_agent_TransportMembership_trmembershipdefault(xmachine_memory_TransportMembership* agent){
	if (h_xmachine_memory_TransportMembership_count + 1 > xmachine_memory_TransportMembership_MAX){
		printf("Error: Buffer size of TransportMembership agents in state trmembershipdefault will be exceeded by h_add_agent_TransportMembership_trmembershipdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_TransportMembership_hostToDevice(d_TransportMemberships_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TransportMembership_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_TransportMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TransportMemberships_trmembershipdefault, d_TransportMemberships_new, h_xmachine_memory_TransportMembership_trmembershipdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_TransportMembership_trmembershipdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TransportMembership_trmembershipdefault_count, &h_xmachine_memory_TransportMembership_trmembershipdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration = 0;
    h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration = 0;
    h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration = 0;
    

}
void h_add_agents_TransportMembership_trmembershipdefault(xmachine_memory_TransportMembership** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_TransportMembership_count + count > xmachine_memory_TransportMembership_MAX){
			printf("Error: Buffer size of TransportMembership agents in state trmembershipdefault will be exceeded by h_add_agents_TransportMembership_trmembershipdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_TransportMembership_AoS_to_SoA(h_TransportMemberships_trmembershipdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_TransportMembership_hostToDevice(d_TransportMemberships_new, h_TransportMemberships_trmembershipdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_TransportMembership_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_TransportMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_TransportMemberships_trmembershipdefault, d_TransportMemberships_new, h_xmachine_memory_TransportMembership_trmembershipdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_TransportMembership_trmembershipdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_TransportMembership_trmembershipdefault_count, &h_xmachine_memory_TransportMembership_trmembershipdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_TransportMemberships_trmembershipdefault_variable_person_id_data_iteration = 0;
        h_TransportMemberships_trmembershipdefault_variable_transport_id_data_iteration = 0;
        h_TransportMemberships_trmembershipdefault_variable_duration_data_iteration = 0;
        

	}
}

xmachine_memory_Clinic* h_allocate_agent_Clinic(){
	xmachine_memory_Clinic* agent = (xmachine_memory_Clinic*)malloc(sizeof(xmachine_memory_Clinic));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Clinic));

	return agent;
}
void h_free_agent_Clinic(xmachine_memory_Clinic** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Clinic** h_allocate_agent_Clinic_array(unsigned int count){
	xmachine_memory_Clinic ** agents = (xmachine_memory_Clinic**)malloc(count * sizeof(xmachine_memory_Clinic*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Clinic();
	}
	return agents;
}
void h_free_agent_Clinic_array(xmachine_memory_Clinic*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Clinic(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Clinic_AoS_to_SoA(xmachine_memory_Clinic_list * dst, xmachine_memory_Clinic** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->lambda[i] = src[i]->lambda;
			
		}
	}
}


void h_add_agent_Clinic_cldefault(xmachine_memory_Clinic* agent){
	if (h_xmachine_memory_Clinic_count + 1 > xmachine_memory_Clinic_MAX){
		printf("Error: Buffer size of Clinic agents in state cldefault will be exceeded by h_add_agent_Clinic_cldefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Clinic_hostToDevice(d_Clinics_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Clinic_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Clinic_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Clinics_cldefault, d_Clinics_new, h_xmachine_memory_Clinic_cldefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Clinic_cldefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Clinic_cldefault_count, &h_xmachine_memory_Clinic_cldefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Clinics_cldefault_variable_id_data_iteration = 0;
    h_Clinics_cldefault_variable_lambda_data_iteration = 0;
    

}
void h_add_agents_Clinic_cldefault(xmachine_memory_Clinic** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Clinic_count + count > xmachine_memory_Clinic_MAX){
			printf("Error: Buffer size of Clinic agents in state cldefault will be exceeded by h_add_agents_Clinic_cldefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Clinic_AoS_to_SoA(h_Clinics_cldefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Clinic_hostToDevice(d_Clinics_new, h_Clinics_cldefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Clinic_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Clinic_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Clinics_cldefault, d_Clinics_new, h_xmachine_memory_Clinic_cldefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Clinic_cldefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Clinic_cldefault_count, &h_xmachine_memory_Clinic_cldefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Clinics_cldefault_variable_id_data_iteration = 0;
        h_Clinics_cldefault_variable_lambda_data_iteration = 0;
        

	}
}

xmachine_memory_Workplace* h_allocate_agent_Workplace(){
	xmachine_memory_Workplace* agent = (xmachine_memory_Workplace*)malloc(sizeof(xmachine_memory_Workplace));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Workplace));

	return agent;
}
void h_free_agent_Workplace(xmachine_memory_Workplace** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Workplace** h_allocate_agent_Workplace_array(unsigned int count){
	xmachine_memory_Workplace ** agents = (xmachine_memory_Workplace**)malloc(count * sizeof(xmachine_memory_Workplace*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Workplace();
	}
	return agents;
}
void h_free_agent_Workplace_array(xmachine_memory_Workplace*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Workplace(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Workplace_AoS_to_SoA(xmachine_memory_Workplace_list * dst, xmachine_memory_Workplace** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->lambda[i] = src[i]->lambda;
			
		}
	}
}


void h_add_agent_Workplace_wpdefault(xmachine_memory_Workplace* agent){
	if (h_xmachine_memory_Workplace_count + 1 > xmachine_memory_Workplace_MAX){
		printf("Error: Buffer size of Workplace agents in state wpdefault will be exceeded by h_add_agent_Workplace_wpdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Workplace_hostToDevice(d_Workplaces_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Workplace_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Workplace_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Workplaces_wpdefault, d_Workplaces_new, h_xmachine_memory_Workplace_wpdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Workplace_wpdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Workplace_wpdefault_count, &h_xmachine_memory_Workplace_wpdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Workplaces_wpdefault_variable_id_data_iteration = 0;
    h_Workplaces_wpdefault_variable_lambda_data_iteration = 0;
    

}
void h_add_agents_Workplace_wpdefault(xmachine_memory_Workplace** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Workplace_count + count > xmachine_memory_Workplace_MAX){
			printf("Error: Buffer size of Workplace agents in state wpdefault will be exceeded by h_add_agents_Workplace_wpdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Workplace_AoS_to_SoA(h_Workplaces_wpdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Workplace_hostToDevice(d_Workplaces_new, h_Workplaces_wpdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Workplace_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Workplace_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Workplaces_wpdefault, d_Workplaces_new, h_xmachine_memory_Workplace_wpdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Workplace_wpdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Workplace_wpdefault_count, &h_xmachine_memory_Workplace_wpdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Workplaces_wpdefault_variable_id_data_iteration = 0;
        h_Workplaces_wpdefault_variable_lambda_data_iteration = 0;
        

	}
}

xmachine_memory_WorkplaceMembership* h_allocate_agent_WorkplaceMembership(){
	xmachine_memory_WorkplaceMembership* agent = (xmachine_memory_WorkplaceMembership*)malloc(sizeof(xmachine_memory_WorkplaceMembership));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_WorkplaceMembership));

	return agent;
}
void h_free_agent_WorkplaceMembership(xmachine_memory_WorkplaceMembership** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_WorkplaceMembership** h_allocate_agent_WorkplaceMembership_array(unsigned int count){
	xmachine_memory_WorkplaceMembership ** agents = (xmachine_memory_WorkplaceMembership**)malloc(count * sizeof(xmachine_memory_WorkplaceMembership*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_WorkplaceMembership();
	}
	return agents;
}
void h_free_agent_WorkplaceMembership_array(xmachine_memory_WorkplaceMembership*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_WorkplaceMembership(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_WorkplaceMembership_AoS_to_SoA(xmachine_memory_WorkplaceMembership_list * dst, xmachine_memory_WorkplaceMembership** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->person_id[i] = src[i]->person_id;
			 
			dst->workplace_id[i] = src[i]->workplace_id;
			
		}
	}
}


void h_add_agent_WorkplaceMembership_wpmembershipdefault(xmachine_memory_WorkplaceMembership* agent){
	if (h_xmachine_memory_WorkplaceMembership_count + 1 > xmachine_memory_WorkplaceMembership_MAX){
		printf("Error: Buffer size of WorkplaceMembership agents in state wpmembershipdefault will be exceeded by h_add_agent_WorkplaceMembership_wpmembershipdefault\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_WorkplaceMembership_hostToDevice(d_WorkplaceMemberships_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_WorkplaceMembership_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_WorkplaceMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_WorkplaceMemberships_wpmembershipdefault, d_WorkplaceMemberships_new, h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, &h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration = 0;
    h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration = 0;
    

}
void h_add_agents_WorkplaceMembership_wpmembershipdefault(xmachine_memory_WorkplaceMembership** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_WorkplaceMembership_count + count > xmachine_memory_WorkplaceMembership_MAX){
			printf("Error: Buffer size of WorkplaceMembership agents in state wpmembershipdefault will be exceeded by h_add_agents_WorkplaceMembership_wpmembershipdefault\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_WorkplaceMembership_AoS_to_SoA(h_WorkplaceMemberships_wpmembershipdefault, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_WorkplaceMembership_hostToDevice(d_WorkplaceMemberships_new, h_WorkplaceMemberships_wpmembershipdefault, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_WorkplaceMembership_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_WorkplaceMembership_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_WorkplaceMemberships_wpmembershipdefault, d_WorkplaceMemberships_new, h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, &h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_WorkplaceMemberships_wpmembershipdefault_variable_person_id_data_iteration = 0;
        h_WorkplaceMemberships_wpmembershipdefault_variable_workplace_id_data_iteration = 0;
        

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
unsigned int reduce_Person_default_householdtime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->householdtime),  thrust::device_pointer_cast(d_Persons_default->householdtime) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_householdtime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->householdtime),  thrust::device_pointer_cast(d_Persons_default->householdtime) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_householdtime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->householdtime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_householdtime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->householdtime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_churchtime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->churchtime),  thrust::device_pointer_cast(d_Persons_default->churchtime) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_churchtime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->churchtime),  thrust::device_pointer_cast(d_Persons_default->churchtime) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_churchtime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchtime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_churchtime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->churchtime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_transporttime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transporttime),  thrust::device_pointer_cast(d_Persons_default->transporttime) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_transporttime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transporttime),  thrust::device_pointer_cast(d_Persons_default->transporttime) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_transporttime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transporttime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_transporttime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transporttime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_clinictime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->clinictime),  thrust::device_pointer_cast(d_Persons_default->clinictime) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_clinictime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->clinictime),  thrust::device_pointer_cast(d_Persons_default->clinictime) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_clinictime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->clinictime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_clinictime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->clinictime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_workplacetime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->workplacetime),  thrust::device_pointer_cast(d_Persons_default->workplacetime) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_workplacetime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->workplacetime),  thrust::device_pointer_cast(d_Persons_default->workplacetime) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_workplacetime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->workplacetime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_workplacetime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->workplacetime);
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
unsigned int reduce_Person_default_transportdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportdur),  thrust::device_pointer_cast(d_Persons_default->transportdur) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_transportdur_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportdur),  thrust::device_pointer_cast(d_Persons_default->transportdur) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_transportdur_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_transportdur_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_transportday1_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportday1),  thrust::device_pointer_cast(d_Persons_default->transportday1) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_transportday1_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportday1),  thrust::device_pointer_cast(d_Persons_default->transportday1) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_transportday1_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportday1);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_transportday1_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportday1);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_transportday2_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transportday2),  thrust::device_pointer_cast(d_Persons_default->transportday2) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_transportday2_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transportday2),  thrust::device_pointer_cast(d_Persons_default->transportday2) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_transportday2_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportday2);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_transportday2_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transportday2);
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
int reduce_Person_default_church_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->church),  thrust::device_pointer_cast(d_Persons_default->church) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_church_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->church),  thrust::device_pointer_cast(d_Persons_default->church) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_church_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->church);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_church_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->church);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_transport_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->transport),  thrust::device_pointer_cast(d_Persons_default->transport) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_transport_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->transport),  thrust::device_pointer_cast(d_Persons_default->transport) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_transport_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transport);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_transport_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->transport);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_workplace_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->workplace),  thrust::device_pointer_cast(d_Persons_default->workplace) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_workplace_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->workplace),  thrust::device_pointer_cast(d_Persons_default->workplace) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_workplace_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->workplace);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_workplace_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->workplace);
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
unsigned int reduce_Person_default_location_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->location),  thrust::device_pointer_cast(d_Persons_default->location) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_location_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->location),  thrust::device_pointer_cast(d_Persons_default->location) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_location_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->location);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_location_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->location);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_locationid_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->locationid),  thrust::device_pointer_cast(d_Persons_default->locationid) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_locationid_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->locationid),  thrust::device_pointer_cast(d_Persons_default->locationid) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_locationid_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->locationid);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_locationid_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->locationid);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_hiv_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->hiv),  thrust::device_pointer_cast(d_Persons_default->hiv) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_hiv_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->hiv),  thrust::device_pointer_cast(d_Persons_default->hiv) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_hiv_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->hiv);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_hiv_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->hiv);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_art_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->art),  thrust::device_pointer_cast(d_Persons_default->art) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_art_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->art),  thrust::device_pointer_cast(d_Persons_default->art) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_art_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->art);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_art_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->art);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_activetb_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->activetb),  thrust::device_pointer_cast(d_Persons_default->activetb) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_activetb_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->activetb),  thrust::device_pointer_cast(d_Persons_default->activetb) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_activetb_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->activetb);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_activetb_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->activetb);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_artday_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->artday),  thrust::device_pointer_cast(d_Persons_default->artday) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_artday_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->artday),  thrust::device_pointer_cast(d_Persons_default->artday) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_artday_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->artday);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_artday_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->artday);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_default_p_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->p),  thrust::device_pointer_cast(d_Persons_default->p) + h_xmachine_memory_Person_default_count);
}

float min_Person_default_p_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->p);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_default_p_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->p);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_default_q_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->q),  thrust::device_pointer_cast(d_Persons_default->q) + h_xmachine_memory_Person_default_count);
}

float min_Person_default_q_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->q);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_default_q_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->q);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_infections_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->infections),  thrust::device_pointer_cast(d_Persons_default->infections) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_infections_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->infections),  thrust::device_pointer_cast(d_Persons_default->infections) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_infections_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->infections);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_infections_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->infections);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_lastinfected_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->lastinfected),  thrust::device_pointer_cast(d_Persons_default->lastinfected) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_lastinfected_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->lastinfected),  thrust::device_pointer_cast(d_Persons_default->lastinfected) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_lastinfected_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lastinfected);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_lastinfected_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lastinfected);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_default_lastinfectedid_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->lastinfectedid),  thrust::device_pointer_cast(d_Persons_default->lastinfectedid) + h_xmachine_memory_Person_default_count);
}

int count_Person_default_lastinfectedid_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->lastinfectedid),  thrust::device_pointer_cast(d_Persons_default->lastinfectedid) + h_xmachine_memory_Person_default_count, count_value);
}
int min_Person_default_lastinfectedid_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lastinfectedid);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_default_lastinfectedid_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lastinfectedid);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_default_time_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->time_step),  thrust::device_pointer_cast(d_Persons_default->time_step) + h_xmachine_memory_Person_default_count);
}

float min_Person_default_time_step_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->time_step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_default_time_step_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->time_step);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_default_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->lambda),  thrust::device_pointer_cast(d_Persons_default->lambda) + h_xmachine_memory_Person_default_count);
}

float min_Person_default_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_default_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_default_timevisiting_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_default->timevisiting),  thrust::device_pointer_cast(d_Persons_default->timevisiting) + h_xmachine_memory_Person_default_count);
}

unsigned int count_Person_default_timevisiting_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_default->timevisiting),  thrust::device_pointer_cast(d_Persons_default->timevisiting) + h_xmachine_memory_Person_default_count, count_value);
}
unsigned int min_Person_default_timevisiting_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->timevisiting);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_default_timevisiting_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_default->timevisiting);
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
unsigned int reduce_Person_s2_householdtime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->householdtime),  thrust::device_pointer_cast(d_Persons_s2->householdtime) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_householdtime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->householdtime),  thrust::device_pointer_cast(d_Persons_s2->householdtime) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_householdtime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->householdtime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_householdtime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->householdtime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_churchtime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->churchtime),  thrust::device_pointer_cast(d_Persons_s2->churchtime) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_churchtime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->churchtime),  thrust::device_pointer_cast(d_Persons_s2->churchtime) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_churchtime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchtime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_churchtime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->churchtime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_transporttime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transporttime),  thrust::device_pointer_cast(d_Persons_s2->transporttime) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_transporttime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transporttime),  thrust::device_pointer_cast(d_Persons_s2->transporttime) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_transporttime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transporttime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_transporttime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transporttime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_clinictime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->clinictime),  thrust::device_pointer_cast(d_Persons_s2->clinictime) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_clinictime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->clinictime),  thrust::device_pointer_cast(d_Persons_s2->clinictime) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_clinictime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->clinictime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_clinictime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->clinictime);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_workplacetime_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->workplacetime),  thrust::device_pointer_cast(d_Persons_s2->workplacetime) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_workplacetime_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->workplacetime),  thrust::device_pointer_cast(d_Persons_s2->workplacetime) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_workplacetime_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->workplacetime);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_workplacetime_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->workplacetime);
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
unsigned int reduce_Person_s2_transportdur_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportdur),  thrust::device_pointer_cast(d_Persons_s2->transportdur) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_transportdur_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportdur),  thrust::device_pointer_cast(d_Persons_s2->transportdur) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_transportdur_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportdur);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_transportdur_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportdur);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_transportday1_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportday1),  thrust::device_pointer_cast(d_Persons_s2->transportday1) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_transportday1_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportday1),  thrust::device_pointer_cast(d_Persons_s2->transportday1) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_transportday1_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportday1);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_transportday1_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportday1);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_transportday2_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transportday2),  thrust::device_pointer_cast(d_Persons_s2->transportday2) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_transportday2_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transportday2),  thrust::device_pointer_cast(d_Persons_s2->transportday2) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_transportday2_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportday2);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_transportday2_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transportday2);
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
int reduce_Person_s2_church_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->church),  thrust::device_pointer_cast(d_Persons_s2->church) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_church_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->church),  thrust::device_pointer_cast(d_Persons_s2->church) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_church_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->church);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_church_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->church);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_transport_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->transport),  thrust::device_pointer_cast(d_Persons_s2->transport) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_transport_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->transport),  thrust::device_pointer_cast(d_Persons_s2->transport) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_transport_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transport);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_transport_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->transport);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_workplace_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->workplace),  thrust::device_pointer_cast(d_Persons_s2->workplace) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_workplace_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->workplace),  thrust::device_pointer_cast(d_Persons_s2->workplace) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_workplace_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->workplace);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_workplace_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->workplace);
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
unsigned int reduce_Person_s2_location_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->location),  thrust::device_pointer_cast(d_Persons_s2->location) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_location_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->location),  thrust::device_pointer_cast(d_Persons_s2->location) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_location_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->location);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_location_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->location);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_locationid_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->locationid),  thrust::device_pointer_cast(d_Persons_s2->locationid) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_locationid_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->locationid),  thrust::device_pointer_cast(d_Persons_s2->locationid) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_locationid_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->locationid);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_locationid_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->locationid);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_hiv_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->hiv),  thrust::device_pointer_cast(d_Persons_s2->hiv) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_hiv_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->hiv),  thrust::device_pointer_cast(d_Persons_s2->hiv) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_hiv_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->hiv);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_hiv_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->hiv);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_art_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->art),  thrust::device_pointer_cast(d_Persons_s2->art) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_art_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->art),  thrust::device_pointer_cast(d_Persons_s2->art) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_art_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->art);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_art_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->art);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_activetb_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->activetb),  thrust::device_pointer_cast(d_Persons_s2->activetb) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_activetb_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->activetb),  thrust::device_pointer_cast(d_Persons_s2->activetb) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_activetb_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->activetb);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_activetb_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->activetb);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_artday_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->artday),  thrust::device_pointer_cast(d_Persons_s2->artday) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_artday_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->artday),  thrust::device_pointer_cast(d_Persons_s2->artday) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_artday_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->artday);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_artday_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->artday);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_s2_p_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->p),  thrust::device_pointer_cast(d_Persons_s2->p) + h_xmachine_memory_Person_s2_count);
}

float min_Person_s2_p_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->p);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_s2_p_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->p);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_s2_q_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->q),  thrust::device_pointer_cast(d_Persons_s2->q) + h_xmachine_memory_Person_s2_count);
}

float min_Person_s2_q_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->q);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_s2_q_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->q);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_infections_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->infections),  thrust::device_pointer_cast(d_Persons_s2->infections) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_infections_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->infections),  thrust::device_pointer_cast(d_Persons_s2->infections) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_infections_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->infections);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_infections_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->infections);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_lastinfected_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->lastinfected),  thrust::device_pointer_cast(d_Persons_s2->lastinfected) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_lastinfected_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->lastinfected),  thrust::device_pointer_cast(d_Persons_s2->lastinfected) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_lastinfected_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lastinfected);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_lastinfected_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lastinfected);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Person_s2_lastinfectedid_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->lastinfectedid),  thrust::device_pointer_cast(d_Persons_s2->lastinfectedid) + h_xmachine_memory_Person_s2_count);
}

int count_Person_s2_lastinfectedid_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->lastinfectedid),  thrust::device_pointer_cast(d_Persons_s2->lastinfectedid) + h_xmachine_memory_Person_s2_count, count_value);
}
int min_Person_s2_lastinfectedid_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lastinfectedid);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Person_s2_lastinfectedid_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lastinfectedid);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_s2_time_step_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->time_step),  thrust::device_pointer_cast(d_Persons_s2->time_step) + h_xmachine_memory_Person_s2_count);
}

float min_Person_s2_time_step_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->time_step);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_s2_time_step_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->time_step);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Person_s2_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->lambda),  thrust::device_pointer_cast(d_Persons_s2->lambda) + h_xmachine_memory_Person_s2_count);
}

float min_Person_s2_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Person_s2_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Person_s2_timevisiting_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Persons_s2->timevisiting),  thrust::device_pointer_cast(d_Persons_s2->timevisiting) + h_xmachine_memory_Person_s2_count);
}

unsigned int count_Person_s2_timevisiting_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Persons_s2->timevisiting),  thrust::device_pointer_cast(d_Persons_s2->timevisiting) + h_xmachine_memory_Person_s2_count, count_value);
}
unsigned int min_Person_s2_timevisiting_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->timevisiting);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Person_s2_timevisiting_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Persons_s2->timevisiting);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Person_s2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_TBAssignment_tbdefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TBAssignments_tbdefault->id),  thrust::device_pointer_cast(d_TBAssignments_tbdefault->id) + h_xmachine_memory_TBAssignment_tbdefault_count);
}

unsigned int count_TBAssignment_tbdefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TBAssignments_tbdefault->id),  thrust::device_pointer_cast(d_TBAssignments_tbdefault->id) + h_xmachine_memory_TBAssignment_tbdefault_count, count_value);
}
unsigned int min_TBAssignment_tbdefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TBAssignments_tbdefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TBAssignment_tbdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_TBAssignment_tbdefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TBAssignments_tbdefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TBAssignment_tbdefault_count) - thrust_ptr;
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
float reduce_Household_hhdefault_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->lambda),  thrust::device_pointer_cast(d_Households_hhdefault->lambda) + h_xmachine_memory_Household_hhdefault_count);
}

float min_Household_hhdefault_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Household_hhdefault_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Household_hhdefault_active_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Households_hhdefault->active),  thrust::device_pointer_cast(d_Households_hhdefault->active) + h_xmachine_memory_Household_hhdefault_count);
}

unsigned int count_Household_hhdefault_active_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Households_hhdefault->active),  thrust::device_pointer_cast(d_Households_hhdefault->active) + h_xmachine_memory_Household_hhdefault_count, count_value);
}
unsigned int min_Household_hhdefault_active_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->active);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Household_hhdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Household_hhdefault_active_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Households_hhdefault->active);
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
unsigned int reduce_HouseholdMembership_hhmembershipdefault_household_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count);
}

unsigned int count_HouseholdMembership_hhmembershipdefault_household_size_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size),  thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size) + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count, count_value);
}
unsigned int min_HouseholdMembership_hhmembershipdefault_household_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_HouseholdMembership_hhmembershipdefault_household_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_HouseholdMemberships_hhmembershipdefault->household_size);
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
float reduce_Church_chudefault_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->lambda),  thrust::device_pointer_cast(d_Churchs_chudefault->lambda) + h_xmachine_memory_Church_chudefault_count);
}

float min_Church_chudefault_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Church_chudefault_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Church_chudefault_active_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Churchs_chudefault->active),  thrust::device_pointer_cast(d_Churchs_chudefault->active) + h_xmachine_memory_Church_chudefault_count);
}

unsigned int count_Church_chudefault_active_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Churchs_chudefault->active),  thrust::device_pointer_cast(d_Churchs_chudefault->active) + h_xmachine_memory_Church_chudefault_count, count_value);
}
unsigned int min_Church_chudefault_active_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->active);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Church_chudefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Church_chudefault_active_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Churchs_chudefault->active);
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
float reduce_Transport_trdefault_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Transports_trdefault->lambda),  thrust::device_pointer_cast(d_Transports_trdefault->lambda) + h_xmachine_memory_Transport_trdefault_count);
}

float min_Transport_trdefault_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Transport_trdefault_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Transport_trdefault_active_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Transports_trdefault->active),  thrust::device_pointer_cast(d_Transports_trdefault->active) + h_xmachine_memory_Transport_trdefault_count);
}

unsigned int count_Transport_trdefault_active_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Transports_trdefault->active),  thrust::device_pointer_cast(d_Transports_trdefault->active) + h_xmachine_memory_Transport_trdefault_count, count_value);
}
unsigned int min_Transport_trdefault_active_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->active);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Transport_trdefault_active_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Transports_trdefault->active);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Transport_trdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_TransportMembership_trmembershipdefault_person_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id) + h_xmachine_memory_TransportMembership_trmembershipdefault_count);
}

int count_TransportMembership_trmembershipdefault_person_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id) + h_xmachine_memory_TransportMembership_trmembershipdefault_count, count_value);
}
int min_TransportMembership_trmembershipdefault_person_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_TransportMembership_trmembershipdefault_person_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->person_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_TransportMembership_trmembershipdefault_transport_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id) + h_xmachine_memory_TransportMembership_trmembershipdefault_count);
}

unsigned int count_TransportMembership_trmembershipdefault_transport_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id) + h_xmachine_memory_TransportMembership_trmembershipdefault_count, count_value);
}
unsigned int min_TransportMembership_trmembershipdefault_transport_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_TransportMembership_trmembershipdefault_transport_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->transport_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_TransportMembership_trmembershipdefault_duration_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration) + h_xmachine_memory_TransportMembership_trmembershipdefault_count);
}

unsigned int count_TransportMembership_trmembershipdefault_duration_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration),  thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration) + h_xmachine_memory_TransportMembership_trmembershipdefault_count, count_value);
}
unsigned int min_TransportMembership_trmembershipdefault_duration_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_TransportMembership_trmembershipdefault_duration_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_TransportMemberships_trmembershipdefault->duration);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_TransportMembership_trmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Clinic_cldefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Clinics_cldefault->id),  thrust::device_pointer_cast(d_Clinics_cldefault->id) + h_xmachine_memory_Clinic_cldefault_count);
}

unsigned int count_Clinic_cldefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Clinics_cldefault->id),  thrust::device_pointer_cast(d_Clinics_cldefault->id) + h_xmachine_memory_Clinic_cldefault_count, count_value);
}
unsigned int min_Clinic_cldefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Clinics_cldefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Clinic_cldefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Clinic_cldefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Clinics_cldefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Clinic_cldefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Clinic_cldefault_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Clinics_cldefault->lambda),  thrust::device_pointer_cast(d_Clinics_cldefault->lambda) + h_xmachine_memory_Clinic_cldefault_count);
}

float min_Clinic_cldefault_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Clinics_cldefault->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Clinic_cldefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Clinic_cldefault_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Clinics_cldefault->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Clinic_cldefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Workplace_wpdefault_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Workplaces_wpdefault->id),  thrust::device_pointer_cast(d_Workplaces_wpdefault->id) + h_xmachine_memory_Workplace_wpdefault_count);
}

unsigned int count_Workplace_wpdefault_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Workplaces_wpdefault->id),  thrust::device_pointer_cast(d_Workplaces_wpdefault->id) + h_xmachine_memory_Workplace_wpdefault_count, count_value);
}
unsigned int min_Workplace_wpdefault_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Workplaces_wpdefault->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Workplace_wpdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Workplace_wpdefault_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Workplaces_wpdefault->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Workplace_wpdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Workplace_wpdefault_lambda_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Workplaces_wpdefault->lambda),  thrust::device_pointer_cast(d_Workplaces_wpdefault->lambda) + h_xmachine_memory_Workplace_wpdefault_count);
}

float min_Workplace_wpdefault_lambda_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Workplaces_wpdefault->lambda);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Workplace_wpdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Workplace_wpdefault_lambda_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Workplaces_wpdefault->lambda);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Workplace_wpdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_WorkplaceMembership_wpmembershipdefault_person_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id),  thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id) + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count);
}

unsigned int count_WorkplaceMembership_wpmembershipdefault_person_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id),  thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id) + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, count_value);
}
unsigned int min_WorkplaceMembership_wpmembershipdefault_person_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_WorkplaceMembership_wpmembershipdefault_person_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->person_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id),  thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id) + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count);
}

unsigned int count_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id),  thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id) + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, count_value);
}
unsigned int min_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_WorkplaceMembership_wpmembershipdefault_workplace_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_WorkplaceMemberships_wpmembershipdefault->workplace_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count) - thrust_ptr;
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

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_Person_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function update\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_update, Person_update_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_update_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_location_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_location_swaps<<<gridSize, blockSize, 0, stream>>>(d_locations); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (update)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_update<<<g, b, sm_size, stream>>>(d_Persons, d_locations, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//location Message Type Prefix Sum
	
	//swap output
	xmachine_message_location_list* d_locations_scanswap_temp = d_locations;
	d_locations = d_locations_swap;
	d_locations_swap = d_locations_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Person, 
        temp_scan_storage_bytes_Person, 
        d_locations_swap->_scan_input,
        d_locations_swap->_position,
        h_xmachine_memory_Person_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_location_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_location_messages<<<gridSize, blockSize, 0, stream>>>(d_locations, d_locations_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_locations_swap->_position[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_locations_swap->_scan_input[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_location_count += scan_last_sum+1;
	}else{
		h_message_location_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of update agents in state s2 will be exceeded moving working agents to next state in function update\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Persons_s2_temp = d_Persons;
  d_Persons = d_Persons_s2;
  d_Persons_s2 = Persons_s2_temp;
        
	//update new state agent size
	h_xmachine_memory_Person_s2_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Person_updatelambda_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_infection));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_updatelambda
 * Agent function prototype for updatelambda function of Person agent
 */
void Person_updatelambda(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_updatelambda, Person_updatelambda_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_updatelambda_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (updatelambda)
	//Reallocate   : false
	//Input        : infection
	//Output       : 
	//Agent Output : 
	GPUFLAME_updatelambda<<<g, b, sm_size, stream>>>(d_Persons, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of updatelambda agents in state s2 will be exceeded moving working agents to next state in function updatelambda\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Persons_s2_temp = d_Persons;
  d_Persons = d_Persons_s2;
  d_Persons_s2 = Persons_s2_temp;
        
	//update new state agent size
	h_xmachine_memory_Person_s2_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Person_infect_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Person_infect
 * Agent function prototype for infect function of Person agent
 */
void Person_infect(cudaStream_t &stream){

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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Person_count = h_xmachine_memory_Person_s2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_count, &h_xmachine_memory_Person_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Person_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Person_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Persons_s2);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Person_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Persons);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, infect_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	infect_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Persons_s2, d_Persons);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Person, 
        temp_scan_storage_bytes_Person, 
        d_Persons_s2->_scan_input,
        d_Persons_s2->_position,
        h_xmachine_memory_Person_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Persons_s2->_position[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Persons_s2->_scan_input[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Person_s2_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Person_s2_count = scan_last_sum;
	//Scatter into swap
	scatter_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_swap, d_Persons_s2, 0, h_xmachine_memory_Person_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Person_list* Persons_s2_temp = d_Persons_s2;
	d_Persons_s2 = d_Persons_swap;
	d_Persons_swap = Persons_s2_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_s2_count, &h_xmachine_memory_Person_s2_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Person, 
        temp_scan_storage_bytes_Person, 
        d_Persons->_scan_input,
        d_Persons->_position,
        h_xmachine_memory_Person_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Persons->_position[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Persons->_scan_input[h_xmachine_memory_Person_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Person_Agents<<<gridSize, blockSize, 0, stream>>>(d_Persons_swap, d_Persons, 0, h_xmachine_memory_Person_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Person_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Person_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Person_list* Persons_temp = d_Persons;
	d_Persons = d_Persons_swap;
	d_Persons_swap = Persons_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_count, &h_xmachine_memory_Person_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Person_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Person_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_infect, Person_infect_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_infect_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (infect)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_infect<<<g, b, sm_size, stream>>>(d_Persons, d_rand48);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of infect agents in state s2 will be exceeded moving working agents to next state in function infect\n");
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
int Person_personhhinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_household_membership));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_personhhinit
 * Agent function prototype for personhhinit function of Person agent
 */
void Person_personhhinit(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_personhhinit, Person_personhhinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_personhhinit_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (personhhinit)
	//Reallocate   : false
	//Input        : household_membership
	//Output       : 
	//Agent Output : 
	GPUFLAME_personhhinit<<<g, b, sm_size, stream>>>(d_Persons, d_household_memberships);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_s2_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of personhhinit agents in state s2 will be exceeded moving working agents to next state in function personhhinit\n");
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
int Person_persontbinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_tb_assignment));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_persontbinit
 * Agent function prototype for persontbinit function of Person agent
 */
void Person_persontbinit(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_persontbinit, Person_persontbinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_persontbinit_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (persontbinit)
	//Reallocate   : false
	//Input        : tb_assignment
	//Output       : 
	//Agent Output : 
	GPUFLAME_persontbinit<<<g, b, sm_size, stream>>>(d_Persons, d_tb_assignments);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_default_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of persontbinit agents in state default will be exceeded moving working agents to next state in function persontbinit\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Persons_default_temp = d_Persons;
  d_Persons = d_Persons_default;
  d_Persons_default = Persons_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Person_default_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Person_persontrinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_transport_membership));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_persontrinit
 * Agent function prototype for persontrinit function of Person agent
 */
void Person_persontrinit(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_persontrinit, Person_persontrinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_persontrinit_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (persontrinit)
	//Reallocate   : false
	//Input        : transport_membership
	//Output       : 
	//Agent Output : 
	GPUFLAME_persontrinit<<<g, b, sm_size, stream>>>(d_Persons, d_transport_memberships);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_default_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of persontrinit agents in state default will be exceeded moving working agents to next state in function persontrinit\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Persons_default_temp = d_Persons;
  d_Persons = d_Persons_default;
  d_Persons_default = Persons_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Person_default_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Person_personwpinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_workplace_membership));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Person_personwpinit
 * Agent function prototype for personwpinit function of Person agent
 */
void Person_personwpinit(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_personwpinit, Person_personwpinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Person_personwpinit_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (personwpinit)
	//Reallocate   : false
	//Input        : workplace_membership
	//Output       : 
	//Agent Output : 
	GPUFLAME_personwpinit<<<g, b, sm_size, stream>>>(d_Persons, d_workplace_memberships);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Person_default_count+h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
		printf("Error: Buffer size of personwpinit agents in state default will be exceeded moving working agents to next state in function personwpinit\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Persons_default_temp = d_Persons;
  d_Persons = d_Persons_default;
  d_Persons_default = Persons_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Person_default_count += h_xmachine_memory_Person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Person_default_count, &h_xmachine_memory_Person_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int TBAssignment_tbinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** TBAssignment_tbinit
 * Agent function prototype for tbinit function of TBAssignment agent
 */
void TBAssignment_tbinit(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_TBAssignment_tbdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_TBAssignment_tbdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_TBAssignment_list* TBAssignments_tbdefault_temp = d_TBAssignments;
	d_TBAssignments = d_TBAssignments_tbdefault;
	d_TBAssignments_tbdefault = TBAssignments_tbdefault_temp;
	//set working count to current state count
	h_xmachine_memory_TBAssignment_count = h_xmachine_memory_TBAssignment_tbdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TBAssignment_count, &h_xmachine_memory_TBAssignment_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_TBAssignment_tbdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TBAssignment_tbdefault_count, &h_xmachine_memory_TBAssignment_tbdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_tb_assignment_count + h_xmachine_memory_TBAssignment_count > xmachine_message_tb_assignment_MAX){
		printf("Error: Buffer size of tb_assignment message will be exceeded in function tbinit\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_tbinit, TBAssignment_tbinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TBAssignment_tbinit_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_tb_assignment_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_tb_assignment_output_type, &h_message_tb_assignment_output_type, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_TBAssignment_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_TBAssignment_scan_input<<<gridSize, blockSize, 0, stream>>>(d_TBAssignments);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (tbinit)
	//Reallocate   : true
	//Input        : 
	//Output       : tb_assignment
	//Agent Output : 
	GPUFLAME_tbinit<<<g, b, sm_size, stream>>>(d_TBAssignments, d_tb_assignments);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_tb_assignment_count += h_xmachine_memory_TBAssignment_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_tb_assignment_count, &h_message_tb_assignment_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TBAssignment, 
        temp_scan_storage_bytes_TBAssignment, 
        d_TBAssignments->_scan_input,
        d_TBAssignments->_position,
        h_xmachine_memory_TBAssignment_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_TBAssignment_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_TBAssignment_Agents<<<gridSize, blockSize, 0, stream>>>(d_TBAssignments_swap, d_TBAssignments, 0, h_xmachine_memory_TBAssignment_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_TBAssignment_list* tbinit_TBAssignments_temp = d_TBAssignments;
	d_TBAssignments = d_TBAssignments_swap;
	d_TBAssignments_swap = tbinit_TBAssignments_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_TBAssignments_swap->_position[h_xmachine_memory_TBAssignment_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_TBAssignments_swap->_scan_input[h_xmachine_memory_TBAssignment_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_TBAssignment_count = scan_last_sum+1;
	else
		h_xmachine_memory_TBAssignment_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TBAssignment_count, &h_xmachine_memory_TBAssignment_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_TBAssignment_tbdefault_count+h_xmachine_memory_TBAssignment_count > xmachine_memory_TBAssignment_MAX){
		printf("Error: Buffer size of tbinit agents in state tbdefault will be exceeded moving working agents to next state in function tbinit\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_TBAssignment_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_TBAssignment_Agents<<<gridSize, blockSize, 0, stream>>>(d_TBAssignments_tbdefault, d_TBAssignments, h_xmachine_memory_TBAssignment_tbdefault_count, h_xmachine_memory_TBAssignment_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_TBAssignment_tbdefault_count += h_xmachine_memory_TBAssignment_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TBAssignment_tbdefault_count, &h_xmachine_memory_TBAssignment_tbdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Household_hhupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Household_count = h_xmachine_memory_Household_hhdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_count, &h_xmachine_memory_Household_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Household_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Household_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Households_hhdefault);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Household_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Households);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hhupdate_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hhupdate_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Households_hhdefault, d_Households);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Household_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Household, 
        temp_scan_storage_bytes_Household, 
        d_Households_hhdefault->_scan_input,
        d_Households_hhdefault->_position,
        h_xmachine_memory_Household_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Households_hhdefault->_position[h_xmachine_memory_Household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Households_hhdefault->_scan_input[h_xmachine_memory_Household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Household_hhdefault_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Household_hhdefault_count = scan_last_sum;
	//Scatter into swap
	scatter_Household_Agents<<<gridSize, blockSize, 0, stream>>>(d_Households_swap, d_Households_hhdefault, 0, h_xmachine_memory_Household_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Household_list* Households_hhdefault_temp = d_Households_hhdefault;
	d_Households_hhdefault = d_Households_swap;
	d_Households_swap = Households_hhdefault_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_hhdefault_count, &h_xmachine_memory_Household_hhdefault_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Household, 
        temp_scan_storage_bytes_Household, 
        d_Households->_scan_input,
        d_Households->_position,
        h_xmachine_memory_Household_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Households->_position[h_xmachine_memory_Household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Households->_scan_input[h_xmachine_memory_Household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Household_Agents<<<gridSize, blockSize, 0, stream>>>(d_Households_swap, d_Households, 0, h_xmachine_memory_Household_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Household_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Household_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Household_list* Households_temp = d_Households;
	d_Households = d_Households_swap;
	d_Households_swap = Households_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Household_count, &h_xmachine_memory_Household_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Household_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Household_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_infection_count + h_xmachine_memory_Household_count > xmachine_message_infection_MAX){
		printf("Error: Buffer size of infection message will be exceeded in function hhupdate\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_hhupdate, Household_hhupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Household_hhupdate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_infection_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_output_type, &h_message_infection_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (hhupdate)
	//Reallocate   : false
	//Input        : location
	//Output       : infection
	//Agent Output : 
	GPUFLAME_hhupdate<<<g, b, sm_size, stream>>>(d_Households, d_locations, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_infection_count += h_xmachine_memory_Household_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Household_hhdefault_count+h_xmachine_memory_Household_count > xmachine_memory_Household_MAX){
		printf("Error: Buffer size of hhupdate agents in state hhdefault will be exceeded moving working agents to next state in function hhupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Household_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Household_Agents<<<gridSize, blockSize, 0, stream>>>(d_Households_hhdefault, d_Households, h_xmachine_memory_Household_hhdefault_count, h_xmachine_memory_Household_count);
  gpuErrchkLaunch();
        
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
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Church_count = h_xmachine_memory_Church_chudefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_count, &h_xmachine_memory_Church_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Church_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Church_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Churchs_chudefault);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Church_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Churchs);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, chuupdate_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	chuupdate_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Churchs_chudefault, d_Churchs);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Church_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Church, 
        temp_scan_storage_bytes_Church, 
        d_Churchs_chudefault->_scan_input,
        d_Churchs_chudefault->_position,
        h_xmachine_memory_Church_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Churchs_chudefault->_position[h_xmachine_memory_Church_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Churchs_chudefault->_scan_input[h_xmachine_memory_Church_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Church_chudefault_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Church_chudefault_count = scan_last_sum;
	//Scatter into swap
	scatter_Church_Agents<<<gridSize, blockSize, 0, stream>>>(d_Churchs_swap, d_Churchs_chudefault, 0, h_xmachine_memory_Church_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Church_list* Churchs_chudefault_temp = d_Churchs_chudefault;
	d_Churchs_chudefault = d_Churchs_swap;
	d_Churchs_swap = Churchs_chudefault_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_chudefault_count, &h_xmachine_memory_Church_chudefault_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Church, 
        temp_scan_storage_bytes_Church, 
        d_Churchs->_scan_input,
        d_Churchs->_position,
        h_xmachine_memory_Church_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Churchs->_position[h_xmachine_memory_Church_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Churchs->_scan_input[h_xmachine_memory_Church_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Church_Agents<<<gridSize, blockSize, 0, stream>>>(d_Churchs_swap, d_Churchs, 0, h_xmachine_memory_Church_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Church_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Church_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Church_list* Churchs_temp = d_Churchs;
	d_Churchs = d_Churchs_swap;
	d_Churchs_swap = Churchs_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Church_count, &h_xmachine_memory_Church_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Church_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Church_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_infection_count + h_xmachine_memory_Church_count > xmachine_message_infection_MAX){
		printf("Error: Buffer size of infection message will be exceeded in function chuupdate\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_chuupdate, Church_chuupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Church_chuupdate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_infection_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_output_type, &h_message_infection_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (chuupdate)
	//Reallocate   : false
	//Input        : location
	//Output       : infection
	//Agent Output : 
	GPUFLAME_chuupdate<<<g, b, sm_size, stream>>>(d_Churchs, d_locations, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_infection_count += h_xmachine_memory_Church_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Church_chudefault_count+h_xmachine_memory_Church_count > xmachine_memory_Church_MAX){
		printf("Error: Buffer size of chuupdate agents in state chudefault will be exceeded moving working agents to next state in function chuupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Church_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Church_Agents<<<gridSize, blockSize, 0, stream>>>(d_Churchs_chudefault, d_Churchs, h_xmachine_memory_Church_chudefault_count, h_xmachine_memory_Church_count);
  gpuErrchkLaunch();
        
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
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Transport_count = h_xmachine_memory_Transport_trdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_count, &h_xmachine_memory_Transport_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Transport_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Transport_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Transports_trdefault);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Transport_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Transports);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, trupdate_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	trupdate_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Transports_trdefault, d_Transports);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Transport_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Transport, 
        temp_scan_storage_bytes_Transport, 
        d_Transports_trdefault->_scan_input,
        d_Transports_trdefault->_position,
        h_xmachine_memory_Transport_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Transports_trdefault->_position[h_xmachine_memory_Transport_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Transports_trdefault->_scan_input[h_xmachine_memory_Transport_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Transport_trdefault_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Transport_trdefault_count = scan_last_sum;
	//Scatter into swap
	scatter_Transport_Agents<<<gridSize, blockSize, 0, stream>>>(d_Transports_swap, d_Transports_trdefault, 0, h_xmachine_memory_Transport_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Transport_list* Transports_trdefault_temp = d_Transports_trdefault;
	d_Transports_trdefault = d_Transports_swap;
	d_Transports_swap = Transports_trdefault_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Transport, 
        temp_scan_storage_bytes_Transport, 
        d_Transports->_scan_input,
        d_Transports->_position,
        h_xmachine_memory_Transport_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Transports->_position[h_xmachine_memory_Transport_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Transports->_scan_input[h_xmachine_memory_Transport_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Transport_Agents<<<gridSize, blockSize, 0, stream>>>(d_Transports_swap, d_Transports, 0, h_xmachine_memory_Transport_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Transport_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Transport_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Transport_list* Transports_temp = d_Transports;
	d_Transports = d_Transports_swap;
	d_Transports_swap = Transports_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_count, &h_xmachine_memory_Transport_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Transport_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Transport_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_infection_count + h_xmachine_memory_Transport_count > xmachine_message_infection_MAX){
		printf("Error: Buffer size of infection message will be exceeded in function trupdate\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_trupdate, Transport_trupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Transport_trupdate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_infection_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_output_type, &h_message_infection_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (trupdate)
	//Reallocate   : false
	//Input        : location
	//Output       : infection
	//Agent Output : 
	GPUFLAME_trupdate<<<g, b, sm_size, stream>>>(d_Transports, d_locations, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_infection_count += h_xmachine_memory_Transport_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Transport_trdefault_count+h_xmachine_memory_Transport_count > xmachine_memory_Transport_MAX){
		printf("Error: Buffer size of trupdate agents in state trdefault will be exceeded moving working agents to next state in function trupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Transport_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Transport_Agents<<<gridSize, blockSize, 0, stream>>>(d_Transports_trdefault, d_Transports, h_xmachine_memory_Transport_trdefault_count, h_xmachine_memory_Transport_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Transport_trdefault_count += h_xmachine_memory_Transport_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Transport_trdefault_count, &h_xmachine_memory_Transport_trdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int TransportMembership_trinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** TransportMembership_trinit
 * Agent function prototype for trinit function of TransportMembership agent
 */
void TransportMembership_trinit(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_TransportMembership_trmembershipdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_TransportMembership_trmembershipdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_TransportMembership_list* TransportMemberships_trmembershipdefault_temp = d_TransportMemberships;
	d_TransportMemberships = d_TransportMemberships_trmembershipdefault;
	d_TransportMemberships_trmembershipdefault = TransportMemberships_trmembershipdefault_temp;
	//set working count to current state count
	h_xmachine_memory_TransportMembership_count = h_xmachine_memory_TransportMembership_trmembershipdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TransportMembership_count, &h_xmachine_memory_TransportMembership_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_TransportMembership_trmembershipdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TransportMembership_trmembershipdefault_count, &h_xmachine_memory_TransportMembership_trmembershipdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_transport_membership_count + h_xmachine_memory_TransportMembership_count > xmachine_message_transport_membership_MAX){
		printf("Error: Buffer size of transport_membership message will be exceeded in function trinit\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_trinit, TransportMembership_trinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = TransportMembership_trinit_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_transport_membership_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_transport_membership_output_type, &h_message_transport_membership_output_type, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_TransportMembership_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_TransportMembership_scan_input<<<gridSize, blockSize, 0, stream>>>(d_TransportMemberships);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (trinit)
	//Reallocate   : true
	//Input        : 
	//Output       : transport_membership
	//Agent Output : 
	GPUFLAME_trinit<<<g, b, sm_size, stream>>>(d_TransportMemberships, d_transport_memberships);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_transport_membership_count += h_xmachine_memory_TransportMembership_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_transport_membership_count, &h_message_transport_membership_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_TransportMembership, 
        temp_scan_storage_bytes_TransportMembership, 
        d_TransportMemberships->_scan_input,
        d_TransportMemberships->_position,
        h_xmachine_memory_TransportMembership_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_TransportMembership_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_TransportMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_TransportMemberships_swap, d_TransportMemberships, 0, h_xmachine_memory_TransportMembership_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_TransportMembership_list* trinit_TransportMemberships_temp = d_TransportMemberships;
	d_TransportMemberships = d_TransportMemberships_swap;
	d_TransportMemberships_swap = trinit_TransportMemberships_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_TransportMemberships_swap->_position[h_xmachine_memory_TransportMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_TransportMemberships_swap->_scan_input[h_xmachine_memory_TransportMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_TransportMembership_count = scan_last_sum+1;
	else
		h_xmachine_memory_TransportMembership_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TransportMembership_count, &h_xmachine_memory_TransportMembership_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_TransportMembership_trmembershipdefault_count+h_xmachine_memory_TransportMembership_count > xmachine_memory_TransportMembership_MAX){
		printf("Error: Buffer size of trinit agents in state trmembershipdefault will be exceeded moving working agents to next state in function trinit\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_TransportMembership_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_TransportMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_TransportMemberships_trmembershipdefault, d_TransportMemberships, h_xmachine_memory_TransportMembership_trmembershipdefault_count, h_xmachine_memory_TransportMembership_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_TransportMembership_trmembershipdefault_count += h_xmachine_memory_TransportMembership_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_TransportMembership_trmembershipdefault_count, &h_xmachine_memory_TransportMembership_trmembershipdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Clinic_clupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Clinic_clupdate
 * Agent function prototype for clupdate function of Clinic agent
 */
void Clinic_clupdate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Clinic_cldefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Clinic_cldefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Clinic_list* Clinics_cldefault_temp = d_Clinics;
	d_Clinics = d_Clinics_cldefault;
	d_Clinics_cldefault = Clinics_cldefault_temp;
	//set working count to current state count
	h_xmachine_memory_Clinic_count = h_xmachine_memory_Clinic_cldefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Clinic_count, &h_xmachine_memory_Clinic_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Clinic_cldefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Clinic_cldefault_count, &h_xmachine_memory_Clinic_cldefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_infection_count + h_xmachine_memory_Clinic_count > xmachine_message_infection_MAX){
		printf("Error: Buffer size of infection message will be exceeded in function clupdate\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_clupdate, Clinic_clupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Clinic_clupdate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_infection_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_output_type, &h_message_infection_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_infection_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_infection_swaps<<<gridSize, blockSize, 0, stream>>>(d_infections); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (clupdate)
	//Reallocate   : false
	//Input        : location
	//Output       : infection
	//Agent Output : 
	GPUFLAME_clupdate<<<g, b, sm_size, stream>>>(d_Clinics, d_locations, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//infection Message Type Prefix Sum
	
	//swap output
	xmachine_message_infection_list* d_infections_scanswap_temp = d_infections;
	d_infections = d_infections_swap;
	d_infections_swap = d_infections_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Clinic, 
        temp_scan_storage_bytes_Clinic, 
        d_infections_swap->_scan_input,
        d_infections_swap->_position,
        h_xmachine_memory_Clinic_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_infection_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_infection_messages<<<gridSize, blockSize, 0, stream>>>(d_infections, d_infections_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_infections_swap->_position[h_xmachine_memory_Clinic_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_infections_swap->_scan_input[h_xmachine_memory_Clinic_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_infection_count += scan_last_sum+1;
	}else{
		h_message_infection_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Clinic_cldefault_count+h_xmachine_memory_Clinic_count > xmachine_memory_Clinic_MAX){
		printf("Error: Buffer size of clupdate agents in state cldefault will be exceeded moving working agents to next state in function clupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Clinics_cldefault_temp = d_Clinics;
  d_Clinics = d_Clinics_cldefault;
  d_Clinics_cldefault = Clinics_cldefault_temp;
        
	//update new state agent size
	h_xmachine_memory_Clinic_cldefault_count += h_xmachine_memory_Clinic_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Clinic_cldefault_count, &h_xmachine_memory_Clinic_cldefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Workplace_wpupdate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Workplace_wpupdate
 * Agent function prototype for wpupdate function of Workplace agent
 */
void Workplace_wpupdate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Workplace_wpdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Workplace_wpdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Workplace_list* Workplaces_wpdefault_temp = d_Workplaces;
	d_Workplaces = d_Workplaces_wpdefault;
	d_Workplaces_wpdefault = Workplaces_wpdefault_temp;
	//set working count to current state count
	h_xmachine_memory_Workplace_count = h_xmachine_memory_Workplace_wpdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Workplace_count, &h_xmachine_memory_Workplace_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Workplace_wpdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Workplace_wpdefault_count, &h_xmachine_memory_Workplace_wpdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_infection_count + h_xmachine_memory_Workplace_count > xmachine_message_infection_MAX){
		printf("Error: Buffer size of infection message will be exceeded in function wpupdate\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_wpupdate, Workplace_wpupdate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Workplace_wpupdate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_infection_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_output_type, &h_message_infection_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_infection_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_infection_swaps<<<gridSize, blockSize, 0, stream>>>(d_infections); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (wpupdate)
	//Reallocate   : false
	//Input        : location
	//Output       : infection
	//Agent Output : 
	GPUFLAME_wpupdate<<<g, b, sm_size, stream>>>(d_Workplaces, d_locations, d_infections);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//infection Message Type Prefix Sum
	
	//swap output
	xmachine_message_infection_list* d_infections_scanswap_temp = d_infections;
	d_infections = d_infections_swap;
	d_infections_swap = d_infections_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Workplace, 
        temp_scan_storage_bytes_Workplace, 
        d_infections_swap->_scan_input,
        d_infections_swap->_position,
        h_xmachine_memory_Workplace_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_infection_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_infection_messages<<<gridSize, blockSize, 0, stream>>>(d_infections, d_infections_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_infections_swap->_position[h_xmachine_memory_Workplace_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_infections_swap->_scan_input[h_xmachine_memory_Workplace_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_infection_count += scan_last_sum+1;
	}else{
		h_message_infection_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_infection_count, &h_message_infection_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Workplace_wpdefault_count+h_xmachine_memory_Workplace_count > xmachine_memory_Workplace_MAX){
		printf("Error: Buffer size of wpupdate agents in state wpdefault will be exceeded moving working agents to next state in function wpupdate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Workplaces_wpdefault_temp = d_Workplaces;
  d_Workplaces = d_Workplaces_wpdefault;
  d_Workplaces_wpdefault = Workplaces_wpdefault_temp;
        
	//update new state agent size
	h_xmachine_memory_Workplace_wpdefault_count += h_xmachine_memory_Workplace_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Workplace_wpdefault_count, &h_xmachine_memory_Workplace_wpdefault_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int WorkplaceMembership_wpinit_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** WorkplaceMembership_wpinit
 * Agent function prototype for wpinit function of WorkplaceMembership agent
 */
void WorkplaceMembership_wpinit(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_WorkplaceMembership_list* WorkplaceMemberships_wpmembershipdefault_temp = d_WorkplaceMemberships;
	d_WorkplaceMemberships = d_WorkplaceMemberships_wpmembershipdefault;
	d_WorkplaceMemberships_wpmembershipdefault = WorkplaceMemberships_wpmembershipdefault_temp;
	//set working count to current state count
	h_xmachine_memory_WorkplaceMembership_count = h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_WorkplaceMembership_count, &h_xmachine_memory_WorkplaceMembership_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, &h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_workplace_membership_count + h_xmachine_memory_WorkplaceMembership_count > xmachine_message_workplace_membership_MAX){
		printf("Error: Buffer size of workplace_membership message will be exceeded in function wpinit\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_wpinit, WorkplaceMembership_wpinit_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = WorkplaceMembership_wpinit_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_workplace_membership_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_workplace_membership_output_type, &h_message_workplace_membership_output_type, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_WorkplaceMembership_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_WorkplaceMembership_scan_input<<<gridSize, blockSize, 0, stream>>>(d_WorkplaceMemberships);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (wpinit)
	//Reallocate   : true
	//Input        : 
	//Output       : workplace_membership
	//Agent Output : 
	GPUFLAME_wpinit<<<g, b, sm_size, stream>>>(d_WorkplaceMemberships, d_workplace_memberships);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_workplace_membership_count += h_xmachine_memory_WorkplaceMembership_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_workplace_membership_count, &h_message_workplace_membership_count, sizeof(int)));	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_WorkplaceMembership, 
        temp_scan_storage_bytes_WorkplaceMembership, 
        d_WorkplaceMemberships->_scan_input,
        d_WorkplaceMemberships->_position,
        h_xmachine_memory_WorkplaceMembership_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_WorkplaceMembership_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_WorkplaceMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_WorkplaceMemberships_swap, d_WorkplaceMemberships, 0, h_xmachine_memory_WorkplaceMembership_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_WorkplaceMembership_list* wpinit_WorkplaceMemberships_temp = d_WorkplaceMemberships;
	d_WorkplaceMemberships = d_WorkplaceMemberships_swap;
	d_WorkplaceMemberships_swap = wpinit_WorkplaceMemberships_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_WorkplaceMemberships_swap->_position[h_xmachine_memory_WorkplaceMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_WorkplaceMemberships_swap->_scan_input[h_xmachine_memory_WorkplaceMembership_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_WorkplaceMembership_count = scan_last_sum+1;
	else
		h_xmachine_memory_WorkplaceMembership_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_WorkplaceMembership_count, &h_xmachine_memory_WorkplaceMembership_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count+h_xmachine_memory_WorkplaceMembership_count > xmachine_memory_WorkplaceMembership_MAX){
		printf("Error: Buffer size of wpinit agents in state wpmembershipdefault will be exceeded moving working agents to next state in function wpinit\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_WorkplaceMembership_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_WorkplaceMembership_Agents<<<gridSize, blockSize, 0, stream>>>(d_WorkplaceMemberships_wpmembershipdefault, d_WorkplaceMemberships, h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, h_xmachine_memory_WorkplaceMembership_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count += h_xmachine_memory_WorkplaceMembership_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, &h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count, sizeof(int)));	
	
	
}


 
extern void reset_Person_default_count()
{
    h_xmachine_memory_Person_default_count = 0;
}
 
extern void reset_Person_s2_count()
{
    h_xmachine_memory_Person_s2_count = 0;
}
 
extern void reset_TBAssignment_tbdefault_count()
{
    h_xmachine_memory_TBAssignment_tbdefault_count = 0;
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
 
extern void reset_TransportMembership_trmembershipdefault_count()
{
    h_xmachine_memory_TransportMembership_trmembershipdefault_count = 0;
}
 
extern void reset_Clinic_cldefault_count()
{
    h_xmachine_memory_Clinic_cldefault_count = 0;
}
 
extern void reset_Workplace_wpdefault_count()
{
    h_xmachine_memory_Workplace_wpdefault_count = 0;
}
 
extern void reset_WorkplaceMembership_wpmembershipdefault_count()
{
    h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count = 0;
}
