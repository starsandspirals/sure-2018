
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>

#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count,xmachine_memory_TBAssignment_list* h_TBAssignments_tbdefault, xmachine_memory_TBAssignment_list* d_TBAssignments_tbdefault, int h_xmachine_memory_TBAssignment_tbdefault_count,xmachine_memory_Household_list* h_Households_hhdefault, xmachine_memory_Household_list* d_Households_hhdefault, int h_xmachine_memory_Household_hhdefault_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships_hhmembershipdefault, xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_hhmembershipdefault, int h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count,xmachine_memory_Church_list* h_Churchs_chudefault, xmachine_memory_Church_list* d_Churchs_chudefault, int h_xmachine_memory_Church_chudefault_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships_chumembershipdefault, xmachine_memory_ChurchMembership_list* d_ChurchMemberships_chumembershipdefault, int h_xmachine_memory_ChurchMembership_chumembershipdefault_count,xmachine_memory_Transport_list* h_Transports_trdefault, xmachine_memory_Transport_list* d_Transports_trdefault, int h_xmachine_memory_Transport_trdefault_count,xmachine_memory_TransportMembership_list* h_TransportMemberships_trmembershipdefault, xmachine_memory_TransportMembership_list* d_TransportMemberships_trmembershipdefault, int h_xmachine_memory_TransportMembership_trmembershipdefault_count,xmachine_memory_Clinic_list* h_Clinics_cldefault, xmachine_memory_Clinic_list* d_Clinics_cldefault, int h_xmachine_memory_Clinic_cldefault_count,xmachine_memory_Workplace_list* h_Workplaces_wpdefault, xmachine_memory_Workplace_list* d_Workplaces_wpdefault, int h_xmachine_memory_Workplace_wpdefault_count,xmachine_memory_WorkplaceMembership_list* h_WorkplaceMemberships_wpmembershipdefault, xmachine_memory_WorkplaceMembership_list* d_WorkplaceMemberships_wpmembershipdefault, int h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count,xmachine_memory_Bar_list* h_Bars_bdefault, xmachine_memory_Bar_list* d_Bars_bdefault, int h_xmachine_memory_Bar_bdefault_count,xmachine_memory_School_list* h_Schools_schdefault, xmachine_memory_School_list* d_Schools_schdefault, int h_xmachine_memory_School_schdefault_count,xmachine_memory_SchoolMembership_list* h_SchoolMemberships_schmembershipdefault, xmachine_memory_SchoolMembership_list* d_SchoolMemberships_schmembershipdefault, int h_xmachine_memory_SchoolMembership_schmembershipdefault_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_Persons_default, d_Persons_default, sizeof(xmachine_memory_Person_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Person Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Persons_s2, d_Persons_s2, sizeof(xmachine_memory_Person_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Person Agent s2 State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_TBAssignments_tbdefault, d_TBAssignments_tbdefault, sizeof(xmachine_memory_TBAssignment_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying TBAssignment Agent tbdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Households_hhdefault, d_Households_hhdefault, sizeof(xmachine_memory_Household_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Household Agent hhdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_HouseholdMemberships_hhmembershipdefault, d_HouseholdMemberships_hhmembershipdefault, sizeof(xmachine_memory_HouseholdMembership_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying HouseholdMembership Agent hhmembershipdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Churchs_chudefault, d_Churchs_chudefault, sizeof(xmachine_memory_Church_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Church Agent chudefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_ChurchMemberships_chumembershipdefault, d_ChurchMemberships_chumembershipdefault, sizeof(xmachine_memory_ChurchMembership_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying ChurchMembership Agent chumembershipdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Transports_trdefault, d_Transports_trdefault, sizeof(xmachine_memory_Transport_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Transport Agent trdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_TransportMemberships_trmembershipdefault, d_TransportMemberships_trmembershipdefault, sizeof(xmachine_memory_TransportMembership_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying TransportMembership Agent trmembershipdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Clinics_cldefault, d_Clinics_cldefault, sizeof(xmachine_memory_Clinic_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Clinic Agent cldefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Workplaces_wpdefault, d_Workplaces_wpdefault, sizeof(xmachine_memory_Workplace_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Workplace Agent wpdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_WorkplaceMemberships_wpmembershipdefault, d_WorkplaceMemberships_wpmembershipdefault, sizeof(xmachine_memory_WorkplaceMembership_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying WorkplaceMembership Agent wpmembershipdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Bars_bdefault, d_Bars_bdefault, sizeof(xmachine_memory_Bar_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Bar Agent bdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_Schools_schdefault, d_Schools_schdefault, sizeof(xmachine_memory_School_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying School Agent schdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_SchoolMemberships_schmembershipdefault, d_SchoolMemberships_schmembershipdefault, sizeof(xmachine_memory_SchoolMembership_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying SchoolMembership Agent schmembershipdefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
    fputs("\t<TIME_STEP>", file);
    sprintf(data, "%f", (*get_TIME_STEP()));
    fputs(data, file);
    fputs("</TIME_STEP>\n", file);
    fputs("\t<MAX_AGE>", file);
    sprintf(data, "%u", (*get_MAX_AGE()));
    fputs(data, file);
    fputs("</MAX_AGE>\n", file);
    fputs("\t<STARTING_POPULATION>", file);
    sprintf(data, "%f", (*get_STARTING_POPULATION()));
    fputs(data, file);
    fputs("</STARTING_POPULATION>\n", file);
    fputs("\t<CHURCH_BETA0>", file);
    sprintf(data, "%f", (*get_CHURCH_BETA0()));
    fputs(data, file);
    fputs("</CHURCH_BETA0>\n", file);
    fputs("\t<CHURCH_BETA1>", file);
    sprintf(data, "%f", (*get_CHURCH_BETA1()));
    fputs(data, file);
    fputs("</CHURCH_BETA1>\n", file);
    fputs("\t<CHURCH_K1>", file);
    sprintf(data, "%u", (*get_CHURCH_K1()));
    fputs(data, file);
    fputs("</CHURCH_K1>\n", file);
    fputs("\t<CHURCH_K2>", file);
    sprintf(data, "%u", (*get_CHURCH_K2()));
    fputs(data, file);
    fputs("</CHURCH_K2>\n", file);
    fputs("\t<CHURCH_K3>", file);
    sprintf(data, "%u", (*get_CHURCH_K3()));
    fputs(data, file);
    fputs("</CHURCH_K3>\n", file);
    fputs("\t<CHURCH_P1>", file);
    sprintf(data, "%f", (*get_CHURCH_P1()));
    fputs(data, file);
    fputs("</CHURCH_P1>\n", file);
    fputs("\t<CHURCH_P2>", file);
    sprintf(data, "%f", (*get_CHURCH_P2()));
    fputs(data, file);
    fputs("</CHURCH_P2>\n", file);
    fputs("\t<CHURCH_PROB0>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB0()));
    fputs(data, file);
    fputs("</CHURCH_PROB0>\n", file);
    fputs("\t<CHURCH_PROB1>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB1()));
    fputs(data, file);
    fputs("</CHURCH_PROB1>\n", file);
    fputs("\t<CHURCH_PROB2>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB2()));
    fputs(data, file);
    fputs("</CHURCH_PROB2>\n", file);
    fputs("\t<CHURCH_PROB3>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB3()));
    fputs(data, file);
    fputs("</CHURCH_PROB3>\n", file);
    fputs("\t<CHURCH_PROB4>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB4()));
    fputs(data, file);
    fputs("</CHURCH_PROB4>\n", file);
    fputs("\t<CHURCH_PROB5>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB5()));
    fputs(data, file);
    fputs("</CHURCH_PROB5>\n", file);
    fputs("\t<CHURCH_PROB6>", file);
    sprintf(data, "%f", (*get_CHURCH_PROB6()));
    fputs(data, file);
    fputs("</CHURCH_PROB6>\n", file);
    fputs("\t<CHURCH_DURATION>", file);
    sprintf(data, "%f", (*get_CHURCH_DURATION()));
    fputs(data, file);
    fputs("</CHURCH_DURATION>\n", file);
    fputs("\t<TRANSPORT_BETA0>", file);
    sprintf(data, "%f", (*get_TRANSPORT_BETA0()));
    fputs(data, file);
    fputs("</TRANSPORT_BETA0>\n", file);
    fputs("\t<TRANSPORT_BETA1>", file);
    sprintf(data, "%f", (*get_TRANSPORT_BETA1()));
    fputs(data, file);
    fputs("</TRANSPORT_BETA1>\n", file);
    fputs("\t<TRANSPORT_FREQ0>", file);
    sprintf(data, "%f", (*get_TRANSPORT_FREQ0()));
    fputs(data, file);
    fputs("</TRANSPORT_FREQ0>\n", file);
    fputs("\t<TRANSPORT_FREQ2>", file);
    sprintf(data, "%f", (*get_TRANSPORT_FREQ2()));
    fputs(data, file);
    fputs("</TRANSPORT_FREQ2>\n", file);
    fputs("\t<TRANSPORT_DUR20>", file);
    sprintf(data, "%f", (*get_TRANSPORT_DUR20()));
    fputs(data, file);
    fputs("</TRANSPORT_DUR20>\n", file);
    fputs("\t<TRANSPORT_DUR45>", file);
    sprintf(data, "%f", (*get_TRANSPORT_DUR45()));
    fputs(data, file);
    fputs("</TRANSPORT_DUR45>\n", file);
    fputs("\t<TRANSPORT_SIZE>", file);
    sprintf(data, "%u", (*get_TRANSPORT_SIZE()));
    fputs(data, file);
    fputs("</TRANSPORT_SIZE>\n", file);
    fputs("\t<HIV_PREVALENCE>", file);
    sprintf(data, "%f", (*get_HIV_PREVALENCE()));
    fputs(data, file);
    fputs("</HIV_PREVALENCE>\n", file);
    fputs("\t<ART_COVERAGE>", file);
    sprintf(data, "%f", (*get_ART_COVERAGE()));
    fputs(data, file);
    fputs("</ART_COVERAGE>\n", file);
    fputs("\t<RR_HIV>", file);
    sprintf(data, "%f", (*get_RR_HIV()));
    fputs(data, file);
    fputs("</RR_HIV>\n", file);
    fputs("\t<RR_ART>", file);
    sprintf(data, "%f", (*get_RR_ART()));
    fputs(data, file);
    fputs("</RR_ART>\n", file);
    fputs("\t<TB_PREVALENCE>", file);
    sprintf(data, "%f", (*get_TB_PREVALENCE()));
    fputs(data, file);
    fputs("</TB_PREVALENCE>\n", file);
    fputs("\t<DEFAULT_P>", file);
    sprintf(data, "%f", (*get_DEFAULT_P()));
    fputs(data, file);
    fputs("</DEFAULT_P>\n", file);
    fputs("\t<DEFAULT_Q>", file);
    sprintf(data, "%f", (*get_DEFAULT_Q()));
    fputs(data, file);
    fputs("</DEFAULT_Q>\n", file);
    fputs("\t<TRANSPORT_A>", file);
    sprintf(data, "%f", (*get_TRANSPORT_A()));
    fputs(data, file);
    fputs("</TRANSPORT_A>\n", file);
    fputs("\t<CHURCH_A>", file);
    sprintf(data, "%f", (*get_CHURCH_A()));
    fputs(data, file);
    fputs("</CHURCH_A>\n", file);
    fputs("\t<CLINIC_A>", file);
    sprintf(data, "%f", (*get_CLINIC_A()));
    fputs(data, file);
    fputs("</CLINIC_A>\n", file);
    fputs("\t<HOUSEHOLD_A>", file);
    sprintf(data, "%f", (*get_HOUSEHOLD_A()));
    fputs(data, file);
    fputs("</HOUSEHOLD_A>\n", file);
    fputs("\t<TRANSPORT_V>", file);
    sprintf(data, "%f", (*get_TRANSPORT_V()));
    fputs(data, file);
    fputs("</TRANSPORT_V>\n", file);
    fputs("\t<HOUSEHOLD_V>", file);
    sprintf(data, "%f", (*get_HOUSEHOLD_V()));
    fputs(data, file);
    fputs("</HOUSEHOLD_V>\n", file);
    fputs("\t<CLINIC_V>", file);
    sprintf(data, "%f", (*get_CLINIC_V()));
    fputs(data, file);
    fputs("</CLINIC_V>\n", file);
    fputs("\t<CHURCH_V_MULTIPLIER>", file);
    sprintf(data, "%f", (*get_CHURCH_V_MULTIPLIER()));
    fputs(data, file);
    fputs("</CHURCH_V_MULTIPLIER>\n", file);
    fputs("\t<WORKPLACE_BETA0>", file);
    sprintf(data, "%f", (*get_WORKPLACE_BETA0()));
    fputs(data, file);
    fputs("</WORKPLACE_BETA0>\n", file);
    fputs("\t<WORKPLACE_BETAA>", file);
    sprintf(data, "%f", (*get_WORKPLACE_BETAA()));
    fputs(data, file);
    fputs("</WORKPLACE_BETAA>\n", file);
    fputs("\t<WORKPLACE_BETAS>", file);
    sprintf(data, "%f", (*get_WORKPLACE_BETAS()));
    fputs(data, file);
    fputs("</WORKPLACE_BETAS>\n", file);
    fputs("\t<WORKPLACE_BETAAS>", file);
    sprintf(data, "%f", (*get_WORKPLACE_BETAAS()));
    fputs(data, file);
    fputs("</WORKPLACE_BETAAS>\n", file);
    fputs("\t<WORKPLACE_A>", file);
    sprintf(data, "%f", (*get_WORKPLACE_A()));
    fputs(data, file);
    fputs("</WORKPLACE_A>\n", file);
    fputs("\t<WORKPLACE_DUR>", file);
    sprintf(data, "%u", (*get_WORKPLACE_DUR()));
    fputs(data, file);
    fputs("</WORKPLACE_DUR>\n", file);
    fputs("\t<WORKPLACE_SIZE>", file);
    sprintf(data, "%u", (*get_WORKPLACE_SIZE()));
    fputs(data, file);
    fputs("</WORKPLACE_SIZE>\n", file);
    fputs("\t<WORKPLACE_V>", file);
    sprintf(data, "%f", (*get_WORKPLACE_V()));
    fputs(data, file);
    fputs("</WORKPLACE_V>\n", file);
    fputs("\t<HOUSEHOLDS>", file);
    sprintf(data, "%u", (*get_HOUSEHOLDS()));
    fputs(data, file);
    fputs("</HOUSEHOLDS>\n", file);
    fputs("\t<BARS>", file);
    sprintf(data, "%u", (*get_BARS()));
    fputs(data, file);
    fputs("</BARS>\n", file);
    fputs("\t<RR_AS_F_46>", file);
    sprintf(data, "%f", (*get_RR_AS_F_46()));
    fputs(data, file);
    fputs("</RR_AS_F_46>\n", file);
    fputs("\t<RR_AS_F_26>", file);
    sprintf(data, "%f", (*get_RR_AS_F_26()));
    fputs(data, file);
    fputs("</RR_AS_F_26>\n", file);
    fputs("\t<RR_AS_F_18>", file);
    sprintf(data, "%f", (*get_RR_AS_F_18()));
    fputs(data, file);
    fputs("</RR_AS_F_18>\n", file);
    fputs("\t<RR_AS_M_46>", file);
    sprintf(data, "%f", (*get_RR_AS_M_46()));
    fputs(data, file);
    fputs("</RR_AS_M_46>\n", file);
    fputs("\t<RR_AS_M_26>", file);
    sprintf(data, "%f", (*get_RR_AS_M_26()));
    fputs(data, file);
    fputs("</RR_AS_M_26>\n", file);
    fputs("\t<RR_AS_M_18>", file);
    sprintf(data, "%f", (*get_RR_AS_M_18()));
    fputs(data, file);
    fputs("</RR_AS_M_18>\n", file);
    fputs("\t<BAR_BETA0>", file);
    sprintf(data, "%f", (*get_BAR_BETA0()));
    fputs(data, file);
    fputs("</BAR_BETA0>\n", file);
    fputs("\t<BAR_BETAA>", file);
    sprintf(data, "%f", (*get_BAR_BETAA()));
    fputs(data, file);
    fputs("</BAR_BETAA>\n", file);
    fputs("\t<BAR_BETAS>", file);
    sprintf(data, "%f", (*get_BAR_BETAS()));
    fputs(data, file);
    fputs("</BAR_BETAS>\n", file);
    fputs("\t<BAR_BETAAS>", file);
    sprintf(data, "%f", (*get_BAR_BETAAS()));
    fputs(data, file);
    fputs("</BAR_BETAAS>\n", file);
    fputs("\t<BAR_SIZE>", file);
    sprintf(data, "%u", (*get_BAR_SIZE()));
    fputs(data, file);
    fputs("</BAR_SIZE>\n", file);
    fputs("\t<SCHOOL_SIZE>", file);
    sprintf(data, "%u", (*get_SCHOOL_SIZE()));
    fputs(data, file);
    fputs("</SCHOOL_SIZE>\n", file);
    fputs("\t<BAR_A>", file);
    sprintf(data, "%f", (*get_BAR_A()));
    fputs(data, file);
    fputs("</BAR_A>\n", file);
    fputs("\t<BAR_V>", file);
    sprintf(data, "%f", (*get_BAR_V()));
    fputs(data, file);
    fputs("</BAR_V>\n", file);
    fputs("\t<SCHOOL_A>", file);
    sprintf(data, "%f", (*get_SCHOOL_A()));
    fputs(data, file);
    fputs("</SCHOOL_A>\n", file);
    fputs("\t<SCHOOL_V>", file);
    sprintf(data, "%f", (*get_SCHOOL_V()));
    fputs(data, file);
    fputs("</SCHOOL_V>\n", file);
    fputs("\t<SEED>", file);
    sprintf(data, "%u", (*get_SEED()));
    fputs(data, file);
    fputs("</SEED>\n", file);
    fputs("\t<HOUSEHOLD_EXP>", file);
    sprintf(data, "%f", (*get_HOUSEHOLD_EXP()));
    fputs(data, file);
    fputs("</HOUSEHOLD_EXP>\n", file);
    fputs("\t<CHURCH_EXP>", file);
    sprintf(data, "%f", (*get_CHURCH_EXP()));
    fputs(data, file);
    fputs("</CHURCH_EXP>\n", file);
    fputs("\t<TRANSPORT_EXP>", file);
    sprintf(data, "%f", (*get_TRANSPORT_EXP()));
    fputs(data, file);
    fputs("</TRANSPORT_EXP>\n", file);
    fputs("\t<CLINIC_EXP>", file);
    sprintf(data, "%f", (*get_CLINIC_EXP()));
    fputs(data, file);
    fputs("</CLINIC_EXP>\n", file);
    fputs("\t<WORKPLACE_EXP>", file);
    sprintf(data, "%f", (*get_WORKPLACE_EXP()));
    fputs(data, file);
    fputs("</WORKPLACE_EXP>\n", file);
    fputs("\t<BAR_EXP>", file);
    sprintf(data, "%f", (*get_BAR_EXP()));
    fputs(data, file);
    fputs("</BAR_EXP>\n", file);
    fputs("\t<SCHOOL_EXP>", file);
    sprintf(data, "%f", (*get_SCHOOL_EXP()));
    fputs(data, file);
    fputs("</SCHOOL_EXP>\n", file);
    fputs("\t<PROB>", file);
    sprintf(data, "%f", (*get_PROB()));
    fputs(data, file);
    fputs("</PROB>\n", file);
    fputs("\t<BAR_M_PROB1>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB1()));
    fputs(data, file);
    fputs("</BAR_M_PROB1>\n", file);
    fputs("\t<BAR_M_PROB2>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB2()));
    fputs(data, file);
    fputs("</BAR_M_PROB2>\n", file);
    fputs("\t<BAR_M_PROB3>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB3()));
    fputs(data, file);
    fputs("</BAR_M_PROB3>\n", file);
    fputs("\t<BAR_M_PROB4>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB4()));
    fputs(data, file);
    fputs("</BAR_M_PROB4>\n", file);
    fputs("\t<BAR_M_PROB5>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB5()));
    fputs(data, file);
    fputs("</BAR_M_PROB5>\n", file);
    fputs("\t<BAR_M_PROB7>", file);
    sprintf(data, "%f", (*get_BAR_M_PROB7()));
    fputs(data, file);
    fputs("</BAR_M_PROB7>\n", file);
    fputs("\t<BAR_F_PROB1>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB1()));
    fputs(data, file);
    fputs("</BAR_F_PROB1>\n", file);
    fputs("\t<BAR_F_PROB2>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB2()));
    fputs(data, file);
    fputs("</BAR_F_PROB2>\n", file);
    fputs("\t<BAR_F_PROB3>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB3()));
    fputs(data, file);
    fputs("</BAR_F_PROB3>\n", file);
    fputs("\t<BAR_F_PROB4>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB4()));
    fputs(data, file);
    fputs("</BAR_F_PROB4>\n", file);
    fputs("\t<BAR_F_PROB5>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB5()));
    fputs(data, file);
    fputs("</BAR_F_PROB5>\n", file);
    fputs("\t<BAR_F_PROB7>", file);
    sprintf(data, "%f", (*get_BAR_F_PROB7()));
    fputs(data, file);
    fputs("</BAR_F_PROB7>\n", file);
	fputs("</environment>\n" , file);

	//Write each Person agent to xml
	for (int i=0; i<h_xmachine_memory_Person_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Person</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Persons_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<step>", file);
        sprintf(data, "%u", h_Persons_default->step[i]);
		fputs(data, file);
		fputs("</step>\n", file);
        
		fputs("<householdtime>", file);
        sprintf(data, "%u", h_Persons_default->householdtime[i]);
		fputs(data, file);
		fputs("</householdtime>\n", file);
        
		fputs("<churchtime>", file);
        sprintf(data, "%u", h_Persons_default->churchtime[i]);
		fputs(data, file);
		fputs("</churchtime>\n", file);
        
		fputs("<transporttime>", file);
        sprintf(data, "%u", h_Persons_default->transporttime[i]);
		fputs(data, file);
		fputs("</transporttime>\n", file);
        
		fputs("<clinictime>", file);
        sprintf(data, "%u", h_Persons_default->clinictime[i]);
		fputs(data, file);
		fputs("</clinictime>\n", file);
        
		fputs("<workplacetime>", file);
        sprintf(data, "%u", h_Persons_default->workplacetime[i]);
		fputs(data, file);
		fputs("</workplacetime>\n", file);
        
		fputs("<bartime>", file);
        sprintf(data, "%u", h_Persons_default->bartime[i]);
		fputs(data, file);
		fputs("</bartime>\n", file);
        
		fputs("<outsidetime>", file);
        sprintf(data, "%u", h_Persons_default->outsidetime[i]);
		fputs(data, file);
		fputs("</outsidetime>\n", file);
        
		fputs("<age>", file);
        sprintf(data, "%u", h_Persons_default->age[i]);
		fputs(data, file);
		fputs("</age>\n", file);
        
		fputs("<gender>", file);
        sprintf(data, "%u", h_Persons_default->gender[i]);
		fputs(data, file);
		fputs("</gender>\n", file);
        
		fputs("<householdsize>", file);
        sprintf(data, "%u", h_Persons_default->householdsize[i]);
		fputs(data, file);
		fputs("</householdsize>\n", file);
        
		fputs("<churchfreq>", file);
        sprintf(data, "%u", h_Persons_default->churchfreq[i]);
		fputs(data, file);
		fputs("</churchfreq>\n", file);
        
		fputs("<churchdur>", file);
        sprintf(data, "%f", h_Persons_default->churchdur[i]);
		fputs(data, file);
		fputs("</churchdur>\n", file);
        
		fputs("<transportdur>", file);
        sprintf(data, "%u", h_Persons_default->transportdur[i]);
		fputs(data, file);
		fputs("</transportdur>\n", file);
        
		fputs("<transportday1>", file);
        sprintf(data, "%d", h_Persons_default->transportday1[i]);
		fputs(data, file);
		fputs("</transportday1>\n", file);
        
		fputs("<transportday2>", file);
        sprintf(data, "%d", h_Persons_default->transportday2[i]);
		fputs(data, file);
		fputs("</transportday2>\n", file);
        
		fputs("<household>", file);
        sprintf(data, "%u", h_Persons_default->household[i]);
		fputs(data, file);
		fputs("</household>\n", file);
        
		fputs("<church>", file);
        sprintf(data, "%d", h_Persons_default->church[i]);
		fputs(data, file);
		fputs("</church>\n", file);
        
		fputs("<transport>", file);
        sprintf(data, "%d", h_Persons_default->transport[i]);
		fputs(data, file);
		fputs("</transport>\n", file);
        
		fputs("<workplace>", file);
        sprintf(data, "%d", h_Persons_default->workplace[i]);
		fputs(data, file);
		fputs("</workplace>\n", file);
        
		fputs("<school>", file);
        sprintf(data, "%d", h_Persons_default->school[i]);
		fputs(data, file);
		fputs("</school>\n", file);
        
		fputs("<busy>", file);
        sprintf(data, "%u", h_Persons_default->busy[i]);
		fputs(data, file);
		fputs("</busy>\n", file);
        
		fputs("<startstep>", file);
        sprintf(data, "%u", h_Persons_default->startstep[i]);
		fputs(data, file);
		fputs("</startstep>\n", file);
        
		fputs("<location>", file);
        sprintf(data, "%u", h_Persons_default->location[i]);
		fputs(data, file);
		fputs("</location>\n", file);
        
		fputs("<locationid>", file);
        sprintf(data, "%u", h_Persons_default->locationid[i]);
		fputs(data, file);
		fputs("</locationid>\n", file);
        
		fputs("<hiv>", file);
        sprintf(data, "%u", h_Persons_default->hiv[i]);
		fputs(data, file);
		fputs("</hiv>\n", file);
        
		fputs("<art>", file);
        sprintf(data, "%u", h_Persons_default->art[i]);
		fputs(data, file);
		fputs("</art>\n", file);
        
		fputs("<activetb>", file);
        sprintf(data, "%u", h_Persons_default->activetb[i]);
		fputs(data, file);
		fputs("</activetb>\n", file);
        
		fputs("<artday>", file);
        sprintf(data, "%u", h_Persons_default->artday[i]);
		fputs(data, file);
		fputs("</artday>\n", file);
        
		fputs("<p>", file);
        sprintf(data, "%f", h_Persons_default->p[i]);
		fputs(data, file);
		fputs("</p>\n", file);
        
		fputs("<q>", file);
        sprintf(data, "%f", h_Persons_default->q[i]);
		fputs(data, file);
		fputs("</q>\n", file);
        
		fputs("<infections>", file);
        sprintf(data, "%u", h_Persons_default->infections[i]);
		fputs(data, file);
		fputs("</infections>\n", file);
        
		fputs("<lastinfected>", file);
        sprintf(data, "%d", h_Persons_default->lastinfected[i]);
		fputs(data, file);
		fputs("</lastinfected>\n", file);
        
		fputs("<lastinfectedid>", file);
        sprintf(data, "%d", h_Persons_default->lastinfectedid[i]);
		fputs(data, file);
		fputs("</lastinfectedid>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Persons_default->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("<timevisiting>", file);
        sprintf(data, "%u", h_Persons_default->timevisiting[i]);
		fputs(data, file);
		fputs("</timevisiting>\n", file);
        
		fputs("<bargoing>", file);
        sprintf(data, "%u", h_Persons_default->bargoing[i]);
		fputs(data, file);
		fputs("</bargoing>\n", file);
        
		fputs("<barday>", file);
        sprintf(data, "%u", h_Persons_default->barday[i]);
		fputs(data, file);
		fputs("</barday>\n", file);
        
		fputs("<schooltime>", file);
        sprintf(data, "%u", h_Persons_default->schooltime[i]);
		fputs(data, file);
		fputs("</schooltime>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Person agent to xml
	for (int i=0; i<h_xmachine_memory_Person_s2_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Person</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Persons_s2->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<step>", file);
        sprintf(data, "%u", h_Persons_s2->step[i]);
		fputs(data, file);
		fputs("</step>\n", file);
        
		fputs("<householdtime>", file);
        sprintf(data, "%u", h_Persons_s2->householdtime[i]);
		fputs(data, file);
		fputs("</householdtime>\n", file);
        
		fputs("<churchtime>", file);
        sprintf(data, "%u", h_Persons_s2->churchtime[i]);
		fputs(data, file);
		fputs("</churchtime>\n", file);
        
		fputs("<transporttime>", file);
        sprintf(data, "%u", h_Persons_s2->transporttime[i]);
		fputs(data, file);
		fputs("</transporttime>\n", file);
        
		fputs("<clinictime>", file);
        sprintf(data, "%u", h_Persons_s2->clinictime[i]);
		fputs(data, file);
		fputs("</clinictime>\n", file);
        
		fputs("<workplacetime>", file);
        sprintf(data, "%u", h_Persons_s2->workplacetime[i]);
		fputs(data, file);
		fputs("</workplacetime>\n", file);
        
		fputs("<bartime>", file);
        sprintf(data, "%u", h_Persons_s2->bartime[i]);
		fputs(data, file);
		fputs("</bartime>\n", file);
        
		fputs("<outsidetime>", file);
        sprintf(data, "%u", h_Persons_s2->outsidetime[i]);
		fputs(data, file);
		fputs("</outsidetime>\n", file);
        
		fputs("<age>", file);
        sprintf(data, "%u", h_Persons_s2->age[i]);
		fputs(data, file);
		fputs("</age>\n", file);
        
		fputs("<gender>", file);
        sprintf(data, "%u", h_Persons_s2->gender[i]);
		fputs(data, file);
		fputs("</gender>\n", file);
        
		fputs("<householdsize>", file);
        sprintf(data, "%u", h_Persons_s2->householdsize[i]);
		fputs(data, file);
		fputs("</householdsize>\n", file);
        
		fputs("<churchfreq>", file);
        sprintf(data, "%u", h_Persons_s2->churchfreq[i]);
		fputs(data, file);
		fputs("</churchfreq>\n", file);
        
		fputs("<churchdur>", file);
        sprintf(data, "%f", h_Persons_s2->churchdur[i]);
		fputs(data, file);
		fputs("</churchdur>\n", file);
        
		fputs("<transportdur>", file);
        sprintf(data, "%u", h_Persons_s2->transportdur[i]);
		fputs(data, file);
		fputs("</transportdur>\n", file);
        
		fputs("<transportday1>", file);
        sprintf(data, "%d", h_Persons_s2->transportday1[i]);
		fputs(data, file);
		fputs("</transportday1>\n", file);
        
		fputs("<transportday2>", file);
        sprintf(data, "%d", h_Persons_s2->transportday2[i]);
		fputs(data, file);
		fputs("</transportday2>\n", file);
        
		fputs("<household>", file);
        sprintf(data, "%u", h_Persons_s2->household[i]);
		fputs(data, file);
		fputs("</household>\n", file);
        
		fputs("<church>", file);
        sprintf(data, "%d", h_Persons_s2->church[i]);
		fputs(data, file);
		fputs("</church>\n", file);
        
		fputs("<transport>", file);
        sprintf(data, "%d", h_Persons_s2->transport[i]);
		fputs(data, file);
		fputs("</transport>\n", file);
        
		fputs("<workplace>", file);
        sprintf(data, "%d", h_Persons_s2->workplace[i]);
		fputs(data, file);
		fputs("</workplace>\n", file);
        
		fputs("<school>", file);
        sprintf(data, "%d", h_Persons_s2->school[i]);
		fputs(data, file);
		fputs("</school>\n", file);
        
		fputs("<busy>", file);
        sprintf(data, "%u", h_Persons_s2->busy[i]);
		fputs(data, file);
		fputs("</busy>\n", file);
        
		fputs("<startstep>", file);
        sprintf(data, "%u", h_Persons_s2->startstep[i]);
		fputs(data, file);
		fputs("</startstep>\n", file);
        
		fputs("<location>", file);
        sprintf(data, "%u", h_Persons_s2->location[i]);
		fputs(data, file);
		fputs("</location>\n", file);
        
		fputs("<locationid>", file);
        sprintf(data, "%u", h_Persons_s2->locationid[i]);
		fputs(data, file);
		fputs("</locationid>\n", file);
        
		fputs("<hiv>", file);
        sprintf(data, "%u", h_Persons_s2->hiv[i]);
		fputs(data, file);
		fputs("</hiv>\n", file);
        
		fputs("<art>", file);
        sprintf(data, "%u", h_Persons_s2->art[i]);
		fputs(data, file);
		fputs("</art>\n", file);
        
		fputs("<activetb>", file);
        sprintf(data, "%u", h_Persons_s2->activetb[i]);
		fputs(data, file);
		fputs("</activetb>\n", file);
        
		fputs("<artday>", file);
        sprintf(data, "%u", h_Persons_s2->artday[i]);
		fputs(data, file);
		fputs("</artday>\n", file);
        
		fputs("<p>", file);
        sprintf(data, "%f", h_Persons_s2->p[i]);
		fputs(data, file);
		fputs("</p>\n", file);
        
		fputs("<q>", file);
        sprintf(data, "%f", h_Persons_s2->q[i]);
		fputs(data, file);
		fputs("</q>\n", file);
        
		fputs("<infections>", file);
        sprintf(data, "%u", h_Persons_s2->infections[i]);
		fputs(data, file);
		fputs("</infections>\n", file);
        
		fputs("<lastinfected>", file);
        sprintf(data, "%d", h_Persons_s2->lastinfected[i]);
		fputs(data, file);
		fputs("</lastinfected>\n", file);
        
		fputs("<lastinfectedid>", file);
        sprintf(data, "%d", h_Persons_s2->lastinfectedid[i]);
		fputs(data, file);
		fputs("</lastinfectedid>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Persons_s2->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("<timevisiting>", file);
        sprintf(data, "%u", h_Persons_s2->timevisiting[i]);
		fputs(data, file);
		fputs("</timevisiting>\n", file);
        
		fputs("<bargoing>", file);
        sprintf(data, "%u", h_Persons_s2->bargoing[i]);
		fputs(data, file);
		fputs("</bargoing>\n", file);
        
		fputs("<barday>", file);
        sprintf(data, "%u", h_Persons_s2->barday[i]);
		fputs(data, file);
		fputs("</barday>\n", file);
        
		fputs("<schooltime>", file);
        sprintf(data, "%u", h_Persons_s2->schooltime[i]);
		fputs(data, file);
		fputs("</schooltime>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each TBAssignment agent to xml
	for (int i=0; i<h_xmachine_memory_TBAssignment_tbdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>TBAssignment</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_TBAssignments_tbdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Household agent to xml
	for (int i=0; i<h_xmachine_memory_Household_hhdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Household</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Households_hhdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Households_hhdefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("<active>", file);
        sprintf(data, "%u", h_Households_hhdefault->active[i]);
		fputs(data, file);
		fputs("</active>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each HouseholdMembership agent to xml
	for (int i=0; i<h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>HouseholdMembership</name>\n", file);
        
		fputs("<household_id>", file);
        sprintf(data, "%u", h_HouseholdMemberships_hhmembershipdefault->household_id[i]);
		fputs(data, file);
		fputs("</household_id>\n", file);
        
		fputs("<person_id>", file);
        sprintf(data, "%u", h_HouseholdMemberships_hhmembershipdefault->person_id[i]);
		fputs(data, file);
		fputs("</person_id>\n", file);
        
		fputs("<household_size>", file);
        sprintf(data, "%u", h_HouseholdMemberships_hhmembershipdefault->household_size[i]);
		fputs(data, file);
		fputs("</household_size>\n", file);
        
		fputs("<churchgoing>", file);
        sprintf(data, "%u", h_HouseholdMemberships_hhmembershipdefault->churchgoing[i]);
		fputs(data, file);
		fputs("</churchgoing>\n", file);
        
		fputs("<churchfreq>", file);
        sprintf(data, "%u", h_HouseholdMemberships_hhmembershipdefault->churchfreq[i]);
		fputs(data, file);
		fputs("</churchfreq>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Church agent to xml
	for (int i=0; i<h_xmachine_memory_Church_chudefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Church</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Churchs_chudefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_Churchs_chudefault->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Churchs_chudefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("<active>", file);
        sprintf(data, "%u", h_Churchs_chudefault->active[i]);
		fputs(data, file);
		fputs("</active>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each ChurchMembership agent to xml
	for (int i=0; i<h_xmachine_memory_ChurchMembership_chumembershipdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>ChurchMembership</name>\n", file);
        
		fputs("<church_id>", file);
        sprintf(data, "%u", h_ChurchMemberships_chumembershipdefault->church_id[i]);
		fputs(data, file);
		fputs("</church_id>\n", file);
        
		fputs("<household_id>", file);
        sprintf(data, "%u", h_ChurchMemberships_chumembershipdefault->household_id[i]);
		fputs(data, file);
		fputs("</household_id>\n", file);
        
		fputs("<churchdur>", file);
        sprintf(data, "%f", h_ChurchMemberships_chumembershipdefault->churchdur[i]);
		fputs(data, file);
		fputs("</churchdur>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Transport agent to xml
	for (int i=0; i<h_xmachine_memory_Transport_trdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Transport</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Transports_trdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Transports_trdefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("<active>", file);
        sprintf(data, "%u", h_Transports_trdefault->active[i]);
		fputs(data, file);
		fputs("</active>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each TransportMembership agent to xml
	for (int i=0; i<h_xmachine_memory_TransportMembership_trmembershipdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>TransportMembership</name>\n", file);
        
		fputs("<person_id>", file);
        sprintf(data, "%d", h_TransportMemberships_trmembershipdefault->person_id[i]);
		fputs(data, file);
		fputs("</person_id>\n", file);
        
		fputs("<transport_id>", file);
        sprintf(data, "%u", h_TransportMemberships_trmembershipdefault->transport_id[i]);
		fputs(data, file);
		fputs("</transport_id>\n", file);
        
		fputs("<duration>", file);
        sprintf(data, "%u", h_TransportMemberships_trmembershipdefault->duration[i]);
		fputs(data, file);
		fputs("</duration>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Clinic agent to xml
	for (int i=0; i<h_xmachine_memory_Clinic_cldefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Clinic</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Clinics_cldefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Clinics_cldefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Workplace agent to xml
	for (int i=0; i<h_xmachine_memory_Workplace_wpdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Workplace</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Workplaces_wpdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Workplaces_wpdefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each WorkplaceMembership agent to xml
	for (int i=0; i<h_xmachine_memory_WorkplaceMembership_wpmembershipdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>WorkplaceMembership</name>\n", file);
        
		fputs("<person_id>", file);
        sprintf(data, "%u", h_WorkplaceMemberships_wpmembershipdefault->person_id[i]);
		fputs(data, file);
		fputs("</person_id>\n", file);
        
		fputs("<workplace_id>", file);
        sprintf(data, "%u", h_WorkplaceMemberships_wpmembershipdefault->workplace_id[i]);
		fputs(data, file);
		fputs("</workplace_id>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each Bar agent to xml
	for (int i=0; i<h_xmachine_memory_Bar_bdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Bar</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Bars_bdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Bars_bdefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each School agent to xml
	for (int i=0; i<h_xmachine_memory_School_schdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>School</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Schools_schdefault->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<lambda>", file);
        sprintf(data, "%f", h_Schools_schdefault->lambda[i]);
		fputs(data, file);
		fputs("</lambda>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each SchoolMembership agent to xml
	for (int i=0; i<h_xmachine_memory_SchoolMembership_schmembershipdefault_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>SchoolMembership</name>\n", file);
        
		fputs("<person_id>", file);
        sprintf(data, "%u", h_SchoolMemberships_schmembershipdefault->person_id[i]);
		fputs(data, file);
		fputs("</person_id>\n", file);
        
		fputs("<school_id>", file);
        sprintf(data, "%u", h_SchoolMemberships_schmembershipdefault->school_id[i]);
		fputs(data, file);
		fputs("</school_id>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void initEnvVars()
{
PROFILE_SCOPED_RANGE("initEnvVars");

    float t_TIME_STEP = (float)1.0;
    set_TIME_STEP(&t_TIME_STEP);
    unsigned int t_MAX_AGE = (unsigned int)100;
    set_MAX_AGE(&t_MAX_AGE);
    float t_STARTING_POPULATION = (float)30000.0;
    set_STARTING_POPULATION(&t_STARTING_POPULATION);
    float t_CHURCH_BETA0 = (float)2.19261;
    set_CHURCH_BETA0(&t_CHURCH_BETA0);
    float t_CHURCH_BETA1 = (float)0.14679;
    set_CHURCH_BETA1(&t_CHURCH_BETA1);
    unsigned int t_CHURCH_K1 = (unsigned int)13;
    set_CHURCH_K1(&t_CHURCH_K1);
    unsigned int t_CHURCH_K2 = (unsigned int)35;
    set_CHURCH_K2(&t_CHURCH_K2);
    unsigned int t_CHURCH_K3 = (unsigned int)100;
    set_CHURCH_K3(&t_CHURCH_K3);
    float t_CHURCH_P1 = (float)0.14;
    set_CHURCH_P1(&t_CHURCH_P1);
    float t_CHURCH_P2 = (float)0.32;
    set_CHURCH_P2(&t_CHURCH_P2);
    float t_CHURCH_PROB0 = (float)0.285569106;
    set_CHURCH_PROB0(&t_CHURCH_PROB0);
    float t_CHURCH_PROB1 = (float)0.704268293;
    set_CHURCH_PROB1(&t_CHURCH_PROB1);
    float t_CHURCH_PROB2 = (float)0.864329269;
    set_CHURCH_PROB2(&t_CHURCH_PROB2);
    float t_CHURCH_PROB3 = (float)0.944613822;
    set_CHURCH_PROB3(&t_CHURCH_PROB3);
    float t_CHURCH_PROB4 = (float)0.978658537;
    set_CHURCH_PROB4(&t_CHURCH_PROB4);
    float t_CHURCH_PROB5 = (float)0.981707317;
    set_CHURCH_PROB5(&t_CHURCH_PROB5);
    float t_CHURCH_PROB6 = (float)0.985772358;
    set_CHURCH_PROB6(&t_CHURCH_PROB6);
    float t_CHURCH_DURATION = (float)0.5;
    set_CHURCH_DURATION(&t_CHURCH_DURATION);
    float t_TRANSPORT_BETA0 = (float)1.682127;
    set_TRANSPORT_BETA0(&t_TRANSPORT_BETA0);
    float t_TRANSPORT_BETA1 = (float)-0.007739;
    set_TRANSPORT_BETA1(&t_TRANSPORT_BETA1);
    float t_TRANSPORT_FREQ0 = (float)0.4337998;
    set_TRANSPORT_FREQ0(&t_TRANSPORT_FREQ0);
    float t_TRANSPORT_FREQ2 = (float)0.8439182;
    set_TRANSPORT_FREQ2(&t_TRANSPORT_FREQ2);
    float t_TRANSPORT_DUR20 = (float)0.5011086;
    set_TRANSPORT_DUR20(&t_TRANSPORT_DUR20);
    float t_TRANSPORT_DUR45 = (float)0.8381374;
    set_TRANSPORT_DUR45(&t_TRANSPORT_DUR45);
    unsigned int t_TRANSPORT_SIZE = (unsigned int)15;
    set_TRANSPORT_SIZE(&t_TRANSPORT_SIZE);
    float t_HIV_PREVALENCE = (float)0.14;
    set_HIV_PREVALENCE(&t_HIV_PREVALENCE);
    float t_ART_COVERAGE = (float)0.21;
    set_ART_COVERAGE(&t_ART_COVERAGE);
    float t_RR_HIV = (float)4.5;
    set_RR_HIV(&t_RR_HIV);
    float t_RR_ART = (float)0.4;
    set_RR_ART(&t_RR_ART);
    float t_TB_PREVALENCE = (float)0.005;
    set_TB_PREVALENCE(&t_TB_PREVALENCE);
    float t_DEFAULT_P = (float)0.36;
    set_DEFAULT_P(&t_DEFAULT_P);
    float t_DEFAULT_Q = (float)1;
    set_DEFAULT_Q(&t_DEFAULT_Q);
    float t_TRANSPORT_A = (float)3;
    set_TRANSPORT_A(&t_TRANSPORT_A);
    float t_CHURCH_A = (float)3;
    set_CHURCH_A(&t_CHURCH_A);
    float t_CLINIC_A = (float)3;
    set_CLINIC_A(&t_CLINIC_A);
    float t_HOUSEHOLD_A = (float)3;
    set_HOUSEHOLD_A(&t_HOUSEHOLD_A);
    float t_TRANSPORT_V = (float)20;
    set_TRANSPORT_V(&t_TRANSPORT_V);
    float t_HOUSEHOLD_V = (float)30;
    set_HOUSEHOLD_V(&t_HOUSEHOLD_V);
    float t_CLINIC_V = (float)40;
    set_CLINIC_V(&t_CLINIC_V);
    float t_CHURCH_V_MULTIPLIER = (float)1;
    set_CHURCH_V_MULTIPLIER(&t_CHURCH_V_MULTIPLIER);
    float t_WORKPLACE_BETA0 = (float)-1.78923;
    set_WORKPLACE_BETA0(&t_WORKPLACE_BETA0);
    float t_WORKPLACE_BETAA = (float)-0.03557;
    set_WORKPLACE_BETAA(&t_WORKPLACE_BETAA);
    float t_WORKPLACE_BETAS = (float)0.16305;
    set_WORKPLACE_BETAS(&t_WORKPLACE_BETAS);
    float t_WORKPLACE_BETAAS = (float)0.04272;
    set_WORKPLACE_BETAAS(&t_WORKPLACE_BETAAS);
    float t_WORKPLACE_A = (float)3;
    set_WORKPLACE_A(&t_WORKPLACE_A);
    unsigned int t_WORKPLACE_DUR = (unsigned int)8;
    set_WORKPLACE_DUR(&t_WORKPLACE_DUR);
    unsigned int t_WORKPLACE_SIZE = (unsigned int)20;
    set_WORKPLACE_SIZE(&t_WORKPLACE_SIZE);
    float t_WORKPLACE_V = (float)40;
    set_WORKPLACE_V(&t_WORKPLACE_V);
    unsigned int t_HOUSEHOLDS = (unsigned int)0;
    set_HOUSEHOLDS(&t_HOUSEHOLDS);
    unsigned int t_BARS = (unsigned int)0;
    set_BARS(&t_BARS);
    float t_RR_AS_F_46 = (float)0.50;
    set_RR_AS_F_46(&t_RR_AS_F_46);
    float t_RR_AS_F_26 = (float)1.25;
    set_RR_AS_F_26(&t_RR_AS_F_26);
    float t_RR_AS_F_18 = (float)1.00;
    set_RR_AS_F_18(&t_RR_AS_F_18);
    float t_RR_AS_M_46 = (float)1.25;
    set_RR_AS_M_46(&t_RR_AS_M_46);
    float t_RR_AS_M_26 = (float)3.75;
    set_RR_AS_M_26(&t_RR_AS_M_26);
    float t_RR_AS_M_18 = (float)1.00;
    set_RR_AS_M_18(&t_RR_AS_M_18);
    float t_BAR_BETA0 = (float)-1.80628;
    set_BAR_BETA0(&t_BAR_BETA0);
    float t_BAR_BETAA = (float)-0.02073;
    set_BAR_BETAA(&t_BAR_BETAA);
    float t_BAR_BETAS = (float)-0.02073;
    set_BAR_BETAS(&t_BAR_BETAS);
    float t_BAR_BETAAS = (float)0.02204;
    set_BAR_BETAAS(&t_BAR_BETAAS);
    unsigned int t_BAR_SIZE = (unsigned int)20;
    set_BAR_SIZE(&t_BAR_SIZE);
    unsigned int t_SCHOOL_SIZE = (unsigned int)40;
    set_SCHOOL_SIZE(&t_SCHOOL_SIZE);
    float t_BAR_A = (float)3;
    set_BAR_A(&t_BAR_A);
    float t_BAR_V = (float)40;
    set_BAR_V(&t_BAR_V);
    float t_SCHOOL_A = (float)3;
    set_SCHOOL_A(&t_SCHOOL_A);
    float t_SCHOOL_V = (float)40;
    set_SCHOOL_V(&t_SCHOOL_V);
    unsigned int t_SEED = (unsigned int)0;
    set_SEED(&t_SEED);
    float t_HOUSEHOLD_EXP = (float)0;
    set_HOUSEHOLD_EXP(&t_HOUSEHOLD_EXP);
    float t_CHURCH_EXP = (float)0;
    set_CHURCH_EXP(&t_CHURCH_EXP);
    float t_TRANSPORT_EXP = (float)0;
    set_TRANSPORT_EXP(&t_TRANSPORT_EXP);
    float t_CLINIC_EXP = (float)0;
    set_CLINIC_EXP(&t_CLINIC_EXP);
    float t_WORKPLACE_EXP = (float)0;
    set_WORKPLACE_EXP(&t_WORKPLACE_EXP);
    float t_BAR_EXP = (float)0;
    set_BAR_EXP(&t_BAR_EXP);
    float t_SCHOOL_EXP = (float)0;
    set_SCHOOL_EXP(&t_SCHOOL_EXP);
    float t_PROB = (float)0;
    set_PROB(&t_PROB);
    float t_BAR_M_PROB1 = (float)0.22;
    set_BAR_M_PROB1(&t_BAR_M_PROB1);
    float t_BAR_M_PROB2 = (float)0.37;
    set_BAR_M_PROB2(&t_BAR_M_PROB2);
    float t_BAR_M_PROB3 = (float)0.51;
    set_BAR_M_PROB3(&t_BAR_M_PROB3);
    float t_BAR_M_PROB4 = (float)0.59;
    set_BAR_M_PROB4(&t_BAR_M_PROB4);
    float t_BAR_M_PROB5 = (float)0.63;
    set_BAR_M_PROB5(&t_BAR_M_PROB5);
    float t_BAR_M_PROB7 = (float)0.74;
    set_BAR_M_PROB7(&t_BAR_M_PROB7);
    float t_BAR_F_PROB1 = (float)0.23;
    set_BAR_F_PROB1(&t_BAR_F_PROB1);
    float t_BAR_F_PROB2 = (float)0.38;
    set_BAR_F_PROB2(&t_BAR_F_PROB2);
    float t_BAR_F_PROB3 = (float)0.55;
    set_BAR_F_PROB3(&t_BAR_F_PROB3);
    float t_BAR_F_PROB4 = (float)0.62;
    set_BAR_F_PROB4(&t_BAR_F_PROB4);
    float t_BAR_F_PROB5 = (float)0.67;
    set_BAR_F_PROB5(&t_BAR_F_PROB5);
    float t_BAR_F_PROB7 = (float)0.71;
    set_BAR_F_PROB7(&t_BAR_F_PROB7);
}

void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count,xmachine_memory_TBAssignment_list* h_TBAssignments, int* h_xmachine_memory_TBAssignment_count,xmachine_memory_Household_list* h_Households, int* h_xmachine_memory_Household_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships, int* h_xmachine_memory_HouseholdMembership_count,xmachine_memory_Church_list* h_Churchs, int* h_xmachine_memory_Church_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships, int* h_xmachine_memory_ChurchMembership_count,xmachine_memory_Transport_list* h_Transports, int* h_xmachine_memory_Transport_count,xmachine_memory_TransportMembership_list* h_TransportMemberships, int* h_xmachine_memory_TransportMembership_count,xmachine_memory_Clinic_list* h_Clinics, int* h_xmachine_memory_Clinic_count,xmachine_memory_Workplace_list* h_Workplaces, int* h_xmachine_memory_Workplace_count,xmachine_memory_WorkplaceMembership_list* h_WorkplaceMemberships, int* h_xmachine_memory_WorkplaceMembership_count,xmachine_memory_Bar_list* h_Bars, int* h_xmachine_memory_Bar_count,xmachine_memory_School_list* h_Schools, int* h_xmachine_memory_School_count,xmachine_memory_SchoolMembership_list* h_SchoolMemberships, int* h_xmachine_memory_SchoolMembership_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name;
    int in_Person_id;
    int in_Person_step;
    int in_Person_householdtime;
    int in_Person_churchtime;
    int in_Person_transporttime;
    int in_Person_clinictime;
    int in_Person_workplacetime;
    int in_Person_bartime;
    int in_Person_outsidetime;
    int in_Person_age;
    int in_Person_gender;
    int in_Person_householdsize;
    int in_Person_churchfreq;
    int in_Person_churchdur;
    int in_Person_transportdur;
    int in_Person_transportday1;
    int in_Person_transportday2;
    int in_Person_household;
    int in_Person_church;
    int in_Person_transport;
    int in_Person_workplace;
    int in_Person_school;
    int in_Person_busy;
    int in_Person_startstep;
    int in_Person_location;
    int in_Person_locationid;
    int in_Person_hiv;
    int in_Person_art;
    int in_Person_activetb;
    int in_Person_artday;
    int in_Person_p;
    int in_Person_q;
    int in_Person_infections;
    int in_Person_lastinfected;
    int in_Person_lastinfectedid;
    int in_Person_lambda;
    int in_Person_timevisiting;
    int in_Person_bargoing;
    int in_Person_barday;
    int in_Person_schooltime;
    int in_TBAssignment_id;
    int in_Household_id;
    int in_Household_lambda;
    int in_Household_active;
    int in_HouseholdMembership_household_id;
    int in_HouseholdMembership_person_id;
    int in_HouseholdMembership_household_size;
    int in_HouseholdMembership_churchgoing;
    int in_HouseholdMembership_churchfreq;
    int in_Church_id;
    int in_Church_size;
    int in_Church_lambda;
    int in_Church_active;
    int in_ChurchMembership_church_id;
    int in_ChurchMembership_household_id;
    int in_ChurchMembership_churchdur;
    int in_Transport_id;
    int in_Transport_lambda;
    int in_Transport_active;
    int in_TransportMembership_person_id;
    int in_TransportMembership_transport_id;
    int in_TransportMembership_duration;
    int in_Clinic_id;
    int in_Clinic_lambda;
    int in_Workplace_id;
    int in_Workplace_lambda;
    int in_WorkplaceMembership_person_id;
    int in_WorkplaceMembership_workplace_id;
    int in_Bar_id;
    int in_Bar_lambda;
    int in_School_id;
    int in_School_lambda;
    int in_SchoolMembership_person_id;
    int in_SchoolMembership_school_id;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_TIME_STEP;
    
    int in_env_MAX_AGE;
    
    int in_env_STARTING_POPULATION;
    
    int in_env_CHURCH_BETA0;
    
    int in_env_CHURCH_BETA1;
    
    int in_env_CHURCH_K1;
    
    int in_env_CHURCH_K2;
    
    int in_env_CHURCH_K3;
    
    int in_env_CHURCH_P1;
    
    int in_env_CHURCH_P2;
    
    int in_env_CHURCH_PROB0;
    
    int in_env_CHURCH_PROB1;
    
    int in_env_CHURCH_PROB2;
    
    int in_env_CHURCH_PROB3;
    
    int in_env_CHURCH_PROB4;
    
    int in_env_CHURCH_PROB5;
    
    int in_env_CHURCH_PROB6;
    
    int in_env_CHURCH_DURATION;
    
    int in_env_TRANSPORT_BETA0;
    
    int in_env_TRANSPORT_BETA1;
    
    int in_env_TRANSPORT_FREQ0;
    
    int in_env_TRANSPORT_FREQ2;
    
    int in_env_TRANSPORT_DUR20;
    
    int in_env_TRANSPORT_DUR45;
    
    int in_env_TRANSPORT_SIZE;
    
    int in_env_HIV_PREVALENCE;
    
    int in_env_ART_COVERAGE;
    
    int in_env_RR_HIV;
    
    int in_env_RR_ART;
    
    int in_env_TB_PREVALENCE;
    
    int in_env_DEFAULT_P;
    
    int in_env_DEFAULT_Q;
    
    int in_env_TRANSPORT_A;
    
    int in_env_CHURCH_A;
    
    int in_env_CLINIC_A;
    
    int in_env_HOUSEHOLD_A;
    
    int in_env_TRANSPORT_V;
    
    int in_env_HOUSEHOLD_V;
    
    int in_env_CLINIC_V;
    
    int in_env_CHURCH_V_MULTIPLIER;
    
    int in_env_WORKPLACE_BETA0;
    
    int in_env_WORKPLACE_BETAA;
    
    int in_env_WORKPLACE_BETAS;
    
    int in_env_WORKPLACE_BETAAS;
    
    int in_env_WORKPLACE_A;
    
    int in_env_WORKPLACE_DUR;
    
    int in_env_WORKPLACE_SIZE;
    
    int in_env_WORKPLACE_V;
    
    int in_env_HOUSEHOLDS;
    
    int in_env_BARS;
    
    int in_env_RR_AS_F_46;
    
    int in_env_RR_AS_F_26;
    
    int in_env_RR_AS_F_18;
    
    int in_env_RR_AS_M_46;
    
    int in_env_RR_AS_M_26;
    
    int in_env_RR_AS_M_18;
    
    int in_env_BAR_BETA0;
    
    int in_env_BAR_BETAA;
    
    int in_env_BAR_BETAS;
    
    int in_env_BAR_BETAAS;
    
    int in_env_BAR_SIZE;
    
    int in_env_SCHOOL_SIZE;
    
    int in_env_BAR_A;
    
    int in_env_BAR_V;
    
    int in_env_SCHOOL_A;
    
    int in_env_SCHOOL_V;
    
    int in_env_SEED;
    
    int in_env_HOUSEHOLD_EXP;
    
    int in_env_CHURCH_EXP;
    
    int in_env_TRANSPORT_EXP;
    
    int in_env_CLINIC_EXP;
    
    int in_env_WORKPLACE_EXP;
    
    int in_env_BAR_EXP;
    
    int in_env_SCHOOL_EXP;
    
    int in_env_PROB;
    
    int in_env_BAR_M_PROB1;
    
    int in_env_BAR_M_PROB2;
    
    int in_env_BAR_M_PROB3;
    
    int in_env_BAR_M_PROB4;
    
    int in_env_BAR_M_PROB5;
    
    int in_env_BAR_M_PROB7;
    
    int in_env_BAR_F_PROB1;
    
    int in_env_BAR_F_PROB2;
    
    int in_env_BAR_F_PROB3;
    
    int in_env_BAR_F_PROB4;
    
    int in_env_BAR_F_PROB5;
    
    int in_env_BAR_F_PROB7;
    
	/* set agent count to zero */
	*h_xmachine_memory_Person_count = 0;
	*h_xmachine_memory_TBAssignment_count = 0;
	*h_xmachine_memory_Household_count = 0;
	*h_xmachine_memory_HouseholdMembership_count = 0;
	*h_xmachine_memory_Church_count = 0;
	*h_xmachine_memory_ChurchMembership_count = 0;
	*h_xmachine_memory_Transport_count = 0;
	*h_xmachine_memory_TransportMembership_count = 0;
	*h_xmachine_memory_Clinic_count = 0;
	*h_xmachine_memory_Workplace_count = 0;
	*h_xmachine_memory_WorkplaceMembership_count = 0;
	*h_xmachine_memory_Bar_count = 0;
	*h_xmachine_memory_School_count = 0;
	*h_xmachine_memory_SchoolMembership_count = 0;
	
	/* Variables for initial state data */
	unsigned int Person_id;
	unsigned int Person_step;
	unsigned int Person_householdtime;
	unsigned int Person_churchtime;
	unsigned int Person_transporttime;
	unsigned int Person_clinictime;
	unsigned int Person_workplacetime;
	unsigned int Person_bartime;
	unsigned int Person_outsidetime;
	unsigned int Person_age;
	unsigned int Person_gender;
	unsigned int Person_householdsize;
	unsigned int Person_churchfreq;
	float Person_churchdur;
	unsigned int Person_transportdur;
	int Person_transportday1;
	int Person_transportday2;
	unsigned int Person_household;
	int Person_church;
	int Person_transport;
	int Person_workplace;
	int Person_school;
	unsigned int Person_busy;
	unsigned int Person_startstep;
	unsigned int Person_location;
	unsigned int Person_locationid;
	unsigned int Person_hiv;
	unsigned int Person_art;
	unsigned int Person_activetb;
	unsigned int Person_artday;
	float Person_p;
	float Person_q;
	unsigned int Person_infections;
	int Person_lastinfected;
	int Person_lastinfectedid;
	float Person_lambda;
	unsigned int Person_timevisiting;
	unsigned int Person_bargoing;
	unsigned int Person_barday;
	unsigned int Person_schooltime;
	unsigned int TBAssignment_id;
	unsigned int Household_id;
	float Household_lambda;
	unsigned int Household_active;
	unsigned int HouseholdMembership_household_id;
	unsigned int HouseholdMembership_person_id;
	unsigned int HouseholdMembership_household_size;
	unsigned int HouseholdMembership_churchgoing;
	unsigned int HouseholdMembership_churchfreq;
	unsigned int Church_id;
	unsigned int Church_size;
	float Church_lambda;
	unsigned int Church_active;
	unsigned int ChurchMembership_church_id;
	unsigned int ChurchMembership_household_id;
	float ChurchMembership_churchdur;
	unsigned int Transport_id;
	float Transport_lambda;
	unsigned int Transport_active;
	int TransportMembership_person_id;
	unsigned int TransportMembership_transport_id;
	unsigned int TransportMembership_duration;
	unsigned int Clinic_id;
	float Clinic_lambda;
	unsigned int Workplace_id;
	float Workplace_lambda;
	unsigned int WorkplaceMembership_person_id;
	unsigned int WorkplaceMembership_workplace_id;
	unsigned int Bar_id;
	float Bar_lambda;
	unsigned int School_id;
	float School_lambda;
	unsigned int SchoolMembership_person_id;
	unsigned int SchoolMembership_school_id;

    /* Variables for environment variables */
    float env_TIME_STEP;
    unsigned int env_MAX_AGE;
    float env_STARTING_POPULATION;
    float env_CHURCH_BETA0;
    float env_CHURCH_BETA1;
    unsigned int env_CHURCH_K1;
    unsigned int env_CHURCH_K2;
    unsigned int env_CHURCH_K3;
    float env_CHURCH_P1;
    float env_CHURCH_P2;
    float env_CHURCH_PROB0;
    float env_CHURCH_PROB1;
    float env_CHURCH_PROB2;
    float env_CHURCH_PROB3;
    float env_CHURCH_PROB4;
    float env_CHURCH_PROB5;
    float env_CHURCH_PROB6;
    float env_CHURCH_DURATION;
    float env_TRANSPORT_BETA0;
    float env_TRANSPORT_BETA1;
    float env_TRANSPORT_FREQ0;
    float env_TRANSPORT_FREQ2;
    float env_TRANSPORT_DUR20;
    float env_TRANSPORT_DUR45;
    unsigned int env_TRANSPORT_SIZE;
    float env_HIV_PREVALENCE;
    float env_ART_COVERAGE;
    float env_RR_HIV;
    float env_RR_ART;
    float env_TB_PREVALENCE;
    float env_DEFAULT_P;
    float env_DEFAULT_Q;
    float env_TRANSPORT_A;
    float env_CHURCH_A;
    float env_CLINIC_A;
    float env_HOUSEHOLD_A;
    float env_TRANSPORT_V;
    float env_HOUSEHOLD_V;
    float env_CLINIC_V;
    float env_CHURCH_V_MULTIPLIER;
    float env_WORKPLACE_BETA0;
    float env_WORKPLACE_BETAA;
    float env_WORKPLACE_BETAS;
    float env_WORKPLACE_BETAAS;
    float env_WORKPLACE_A;
    unsigned int env_WORKPLACE_DUR;
    unsigned int env_WORKPLACE_SIZE;
    float env_WORKPLACE_V;
    unsigned int env_HOUSEHOLDS;
    unsigned int env_BARS;
    float env_RR_AS_F_46;
    float env_RR_AS_F_26;
    float env_RR_AS_F_18;
    float env_RR_AS_M_46;
    float env_RR_AS_M_26;
    float env_RR_AS_M_18;
    float env_BAR_BETA0;
    float env_BAR_BETAA;
    float env_BAR_BETAS;
    float env_BAR_BETAAS;
    unsigned int env_BAR_SIZE;
    unsigned int env_SCHOOL_SIZE;
    float env_BAR_A;
    float env_BAR_V;
    float env_SCHOOL_A;
    float env_SCHOOL_V;
    unsigned int env_SEED;
    float env_HOUSEHOLD_EXP;
    float env_CHURCH_EXP;
    float env_TRANSPORT_EXP;
    float env_CLINIC_EXP;
    float env_WORKPLACE_EXP;
    float env_BAR_EXP;
    float env_SCHOOL_EXP;
    float env_PROB;
    float env_BAR_M_PROB1;
    float env_BAR_M_PROB2;
    float env_BAR_M_PROB3;
    float env_BAR_M_PROB4;
    float env_BAR_M_PROB5;
    float env_BAR_M_PROB7;
    float env_BAR_F_PROB1;
    float env_BAR_F_PROB2;
    float env_BAR_F_PROB3;
    float env_BAR_F_PROB4;
    float env_BAR_F_PROB5;
    float env_BAR_F_PROB7;
    


	/* Initialise variables */
    initEnvVars();
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_Person_id = 0;
	in_Person_step = 0;
	in_Person_householdtime = 0;
	in_Person_churchtime = 0;
	in_Person_transporttime = 0;
	in_Person_clinictime = 0;
	in_Person_workplacetime = 0;
	in_Person_bartime = 0;
	in_Person_outsidetime = 0;
	in_Person_age = 0;
	in_Person_gender = 0;
	in_Person_householdsize = 0;
	in_Person_churchfreq = 0;
	in_Person_churchdur = 0;
	in_Person_transportdur = 0;
	in_Person_transportday1 = 0;
	in_Person_transportday2 = 0;
	in_Person_household = 0;
	in_Person_church = 0;
	in_Person_transport = 0;
	in_Person_workplace = 0;
	in_Person_school = 0;
	in_Person_busy = 0;
	in_Person_startstep = 0;
	in_Person_location = 0;
	in_Person_locationid = 0;
	in_Person_hiv = 0;
	in_Person_art = 0;
	in_Person_activetb = 0;
	in_Person_artday = 0;
	in_Person_p = 0;
	in_Person_q = 0;
	in_Person_infections = 0;
	in_Person_lastinfected = 0;
	in_Person_lastinfectedid = 0;
	in_Person_lambda = 0;
	in_Person_timevisiting = 0;
	in_Person_bargoing = 0;
	in_Person_barday = 0;
	in_Person_schooltime = 0;
	in_TBAssignment_id = 0;
	in_Household_id = 0;
	in_Household_lambda = 0;
	in_Household_active = 0;
	in_HouseholdMembership_household_id = 0;
	in_HouseholdMembership_person_id = 0;
	in_HouseholdMembership_household_size = 0;
	in_HouseholdMembership_churchgoing = 0;
	in_HouseholdMembership_churchfreq = 0;
	in_Church_id = 0;
	in_Church_size = 0;
	in_Church_lambda = 0;
	in_Church_active = 0;
	in_ChurchMembership_church_id = 0;
	in_ChurchMembership_household_id = 0;
	in_ChurchMembership_churchdur = 0;
	in_Transport_id = 0;
	in_Transport_lambda = 0;
	in_Transport_active = 0;
	in_TransportMembership_person_id = 0;
	in_TransportMembership_transport_id = 0;
	in_TransportMembership_duration = 0;
	in_Clinic_id = 0;
	in_Clinic_lambda = 0;
	in_Workplace_id = 0;
	in_Workplace_lambda = 0;
	in_WorkplaceMembership_person_id = 0;
	in_WorkplaceMembership_workplace_id = 0;
	in_Bar_id = 0;
	in_Bar_lambda = 0;
	in_School_id = 0;
	in_School_lambda = 0;
	in_SchoolMembership_person_id = 0;
	in_SchoolMembership_school_id = 0;
    in_env_TIME_STEP = 0;
    in_env_MAX_AGE = 0;
    in_env_STARTING_POPULATION = 0;
    in_env_CHURCH_BETA0 = 0;
    in_env_CHURCH_BETA1 = 0;
    in_env_CHURCH_K1 = 0;
    in_env_CHURCH_K2 = 0;
    in_env_CHURCH_K3 = 0;
    in_env_CHURCH_P1 = 0;
    in_env_CHURCH_P2 = 0;
    in_env_CHURCH_PROB0 = 0;
    in_env_CHURCH_PROB1 = 0;
    in_env_CHURCH_PROB2 = 0;
    in_env_CHURCH_PROB3 = 0;
    in_env_CHURCH_PROB4 = 0;
    in_env_CHURCH_PROB5 = 0;
    in_env_CHURCH_PROB6 = 0;
    in_env_CHURCH_DURATION = 0;
    in_env_TRANSPORT_BETA0 = 0;
    in_env_TRANSPORT_BETA1 = 0;
    in_env_TRANSPORT_FREQ0 = 0;
    in_env_TRANSPORT_FREQ2 = 0;
    in_env_TRANSPORT_DUR20 = 0;
    in_env_TRANSPORT_DUR45 = 0;
    in_env_TRANSPORT_SIZE = 0;
    in_env_HIV_PREVALENCE = 0;
    in_env_ART_COVERAGE = 0;
    in_env_RR_HIV = 0;
    in_env_RR_ART = 0;
    in_env_TB_PREVALENCE = 0;
    in_env_DEFAULT_P = 0;
    in_env_DEFAULT_Q = 0;
    in_env_TRANSPORT_A = 0;
    in_env_CHURCH_A = 0;
    in_env_CLINIC_A = 0;
    in_env_HOUSEHOLD_A = 0;
    in_env_TRANSPORT_V = 0;
    in_env_HOUSEHOLD_V = 0;
    in_env_CLINIC_V = 0;
    in_env_CHURCH_V_MULTIPLIER = 0;
    in_env_WORKPLACE_BETA0 = 0;
    in_env_WORKPLACE_BETAA = 0;
    in_env_WORKPLACE_BETAS = 0;
    in_env_WORKPLACE_BETAAS = 0;
    in_env_WORKPLACE_A = 0;
    in_env_WORKPLACE_DUR = 0;
    in_env_WORKPLACE_SIZE = 0;
    in_env_WORKPLACE_V = 0;
    in_env_HOUSEHOLDS = 0;
    in_env_BARS = 0;
    in_env_RR_AS_F_46 = 0;
    in_env_RR_AS_F_26 = 0;
    in_env_RR_AS_F_18 = 0;
    in_env_RR_AS_M_46 = 0;
    in_env_RR_AS_M_26 = 0;
    in_env_RR_AS_M_18 = 0;
    in_env_BAR_BETA0 = 0;
    in_env_BAR_BETAA = 0;
    in_env_BAR_BETAS = 0;
    in_env_BAR_BETAAS = 0;
    in_env_BAR_SIZE = 0;
    in_env_SCHOOL_SIZE = 0;
    in_env_BAR_A = 0;
    in_env_BAR_V = 0;
    in_env_SCHOOL_A = 0;
    in_env_SCHOOL_V = 0;
    in_env_SEED = 0;
    in_env_HOUSEHOLD_EXP = 0;
    in_env_CHURCH_EXP = 0;
    in_env_TRANSPORT_EXP = 0;
    in_env_CLINIC_EXP = 0;
    in_env_WORKPLACE_EXP = 0;
    in_env_BAR_EXP = 0;
    in_env_SCHOOL_EXP = 0;
    in_env_PROB = 0;
    in_env_BAR_M_PROB1 = 0;
    in_env_BAR_M_PROB2 = 0;
    in_env_BAR_M_PROB3 = 0;
    in_env_BAR_M_PROB4 = 0;
    in_env_BAR_M_PROB5 = 0;
    in_env_BAR_M_PROB7 = 0;
    in_env_BAR_F_PROB1 = 0;
    in_env_BAR_F_PROB2 = 0;
    in_env_BAR_F_PROB3 = 0;
    in_env_BAR_F_PROB4 = 0;
    in_env_BAR_F_PROB5 = 0;
    in_env_BAR_F_PROB7 = 0;
	//set all Person values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Person_MAX; k++)
	{	
		h_Persons->id[k] = 0;
		h_Persons->step[k] = 0;
		h_Persons->householdtime[k] = 0;
		h_Persons->churchtime[k] = 0;
		h_Persons->transporttime[k] = 0;
		h_Persons->clinictime[k] = 0;
		h_Persons->workplacetime[k] = 0;
		h_Persons->bartime[k] = 0;
		h_Persons->outsidetime[k] = 0;
		h_Persons->age[k] = 0;
		h_Persons->gender[k] = 0;
		h_Persons->householdsize[k] = 0;
		h_Persons->churchfreq[k] = 0;
		h_Persons->churchdur[k] = 0;
		h_Persons->transportdur[k] = 0;
		h_Persons->transportday1[k] = 0;
		h_Persons->transportday2[k] = 0;
		h_Persons->household[k] = 0;
		h_Persons->church[k] = 0;
		h_Persons->transport[k] = 0;
		h_Persons->workplace[k] = 0;
		h_Persons->school[k] = 0;
		h_Persons->busy[k] = 0;
		h_Persons->startstep[k] = 0;
		h_Persons->location[k] = 0;
		h_Persons->locationid[k] = 0;
		h_Persons->hiv[k] = 0;
		h_Persons->art[k] = 0;
		h_Persons->activetb[k] = 0;
		h_Persons->artday[k] = 0;
		h_Persons->p[k] = 0;
		h_Persons->q[k] = 0;
		h_Persons->infections[k] = 0;
		h_Persons->lastinfected[k] = 0;
		h_Persons->lastinfectedid[k] = 0;
		h_Persons->lambda[k] = 0;
		h_Persons->timevisiting[k] = 0;
		h_Persons->bargoing[k] = 0;
		h_Persons->barday[k] = 0;
		h_Persons->schooltime[k] = 0;
	}
	
	//set all TBAssignment values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_TBAssignment_MAX; k++)
	{	
		h_TBAssignments->id[k] = 0;
	}
	
	//set all Household values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Household_MAX; k++)
	{	
		h_Households->id[k] = 0;
		h_Households->lambda[k] = 0;
		h_Households->active[k] = 0;
	}
	
	//set all HouseholdMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_HouseholdMembership_MAX; k++)
	{	
		h_HouseholdMemberships->household_id[k] = 0;
		h_HouseholdMemberships->person_id[k] = 0;
		h_HouseholdMemberships->household_size[k] = 0;
		h_HouseholdMemberships->churchgoing[k] = 0;
		h_HouseholdMemberships->churchfreq[k] = 0;
	}
	
	//set all Church values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Church_MAX; k++)
	{	
		h_Churchs->id[k] = 0;
		h_Churchs->size[k] = 0;
		h_Churchs->lambda[k] = 0;
		h_Churchs->active[k] = 0;
	}
	
	//set all ChurchMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_ChurchMembership_MAX; k++)
	{	
		h_ChurchMemberships->church_id[k] = 0;
		h_ChurchMemberships->household_id[k] = 0;
		h_ChurchMemberships->churchdur[k] = 0;
	}
	
	//set all Transport values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Transport_MAX; k++)
	{	
		h_Transports->id[k] = 0;
		h_Transports->lambda[k] = 0;
		h_Transports->active[k] = 0;
	}
	
	//set all TransportMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_TransportMembership_MAX; k++)
	{	
		h_TransportMemberships->person_id[k] = 0;
		h_TransportMemberships->transport_id[k] = 0;
		h_TransportMemberships->duration[k] = 0;
	}
	
	//set all Clinic values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Clinic_MAX; k++)
	{	
		h_Clinics->id[k] = 0;
		h_Clinics->lambda[k] = 0;
	}
	
	//set all Workplace values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Workplace_MAX; k++)
	{	
		h_Workplaces->id[k] = 0;
		h_Workplaces->lambda[k] = 0;
	}
	
	//set all WorkplaceMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_WorkplaceMembership_MAX; k++)
	{	
		h_WorkplaceMemberships->person_id[k] = 0;
		h_WorkplaceMemberships->workplace_id[k] = 0;
	}
	
	//set all Bar values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Bar_MAX; k++)
	{	
		h_Bars->id[k] = 0;
		h_Bars->lambda[k] = 0;
	}
	
	//set all School values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_School_MAX; k++)
	{	
		h_Schools->id[k] = 0;
		h_Schools->lambda[k] = 0;
	}
	
	//set all SchoolMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_SchoolMembership_MAX; k++)
	{	
		h_SchoolMemberships->person_id[k] = 0;
		h_SchoolMemberships->school_id[k] = 0;
	}
	

	/* Default variables for memory */
    Person_id = 0;
    Person_step = 0;
    Person_householdtime = 0;
    Person_churchtime = 0;
    Person_transporttime = 0;
    Person_clinictime = 0;
    Person_workplacetime = 0;
    Person_bartime = 0;
    Person_outsidetime = 0;
    Person_age = 0;
    Person_gender = 0;
    Person_householdsize = 0;
    Person_churchfreq = 0;
    Person_churchdur = 0;
    Person_transportdur = 0;
    Person_transportday1 = 0;
    Person_transportday2 = 0;
    Person_household = 0;
    Person_church = 0;
    Person_transport = 0;
    Person_workplace = 0;
    Person_school = 0;
    Person_busy = 0;
    Person_startstep = 0;
    Person_location = 0;
    Person_locationid = 0;
    Person_hiv = 0;
    Person_art = 0;
    Person_activetb = 0;
    Person_artday = 0;
    Person_p = 0;
    Person_q = 0;
    Person_infections = 0;
    Person_lastinfected = 0;
    Person_lastinfectedid = 0;
    Person_lambda = 0;
    Person_timevisiting = 0;
    Person_bargoing = 0;
    Person_barday = 0;
    Person_schooltime = 0;
    TBAssignment_id = 0;
    Household_id = 0;
    Household_lambda = 0;
    Household_active = 0;
    HouseholdMembership_household_id = 0;
    HouseholdMembership_person_id = 0;
    HouseholdMembership_household_size = 0;
    HouseholdMembership_churchgoing = 0;
    HouseholdMembership_churchfreq = 0;
    Church_id = 0;
    Church_size = 0;
    Church_lambda = 0;
    Church_active = 0;
    ChurchMembership_church_id = 0;
    ChurchMembership_household_id = 0;
    ChurchMembership_churchdur = 0;
    Transport_id = 0;
    Transport_lambda = 0;
    Transport_active = 0;
    TransportMembership_person_id = 0;
    TransportMembership_transport_id = 0;
    TransportMembership_duration = 0;
    Clinic_id = 0;
    Clinic_lambda = 0;
    Workplace_id = 0;
    Workplace_lambda = 0;
    WorkplaceMembership_person_id = 0;
    WorkplaceMembership_workplace_id = 0;
    Bar_id = 0;
    Bar_lambda = 0;
    School_id = 0;
    School_lambda = 0;
    SchoolMembership_person_id = 0;
    SchoolMembership_school_id = 0;

    /* Default variables for environment variables */
    env_TIME_STEP = 0;
    env_MAX_AGE = 0;
    env_STARTING_POPULATION = 0;
    env_CHURCH_BETA0 = 0;
    env_CHURCH_BETA1 = 0;
    env_CHURCH_K1 = 0;
    env_CHURCH_K2 = 0;
    env_CHURCH_K3 = 0;
    env_CHURCH_P1 = 0;
    env_CHURCH_P2 = 0;
    env_CHURCH_PROB0 = 0;
    env_CHURCH_PROB1 = 0;
    env_CHURCH_PROB2 = 0;
    env_CHURCH_PROB3 = 0;
    env_CHURCH_PROB4 = 0;
    env_CHURCH_PROB5 = 0;
    env_CHURCH_PROB6 = 0;
    env_CHURCH_DURATION = 0;
    env_TRANSPORT_BETA0 = 0;
    env_TRANSPORT_BETA1 = 0;
    env_TRANSPORT_FREQ0 = 0;
    env_TRANSPORT_FREQ2 = 0;
    env_TRANSPORT_DUR20 = 0;
    env_TRANSPORT_DUR45 = 0;
    env_TRANSPORT_SIZE = 0;
    env_HIV_PREVALENCE = 0;
    env_ART_COVERAGE = 0;
    env_RR_HIV = 0;
    env_RR_ART = 0;
    env_TB_PREVALENCE = 0;
    env_DEFAULT_P = 0;
    env_DEFAULT_Q = 0;
    env_TRANSPORT_A = 0;
    env_CHURCH_A = 0;
    env_CLINIC_A = 0;
    env_HOUSEHOLD_A = 0;
    env_TRANSPORT_V = 0;
    env_HOUSEHOLD_V = 0;
    env_CLINIC_V = 0;
    env_CHURCH_V_MULTIPLIER = 0;
    env_WORKPLACE_BETA0 = 0;
    env_WORKPLACE_BETAA = 0;
    env_WORKPLACE_BETAS = 0;
    env_WORKPLACE_BETAAS = 0;
    env_WORKPLACE_A = 0;
    env_WORKPLACE_DUR = 0;
    env_WORKPLACE_SIZE = 0;
    env_WORKPLACE_V = 0;
    env_HOUSEHOLDS = 0;
    env_BARS = 0;
    env_RR_AS_F_46 = 0;
    env_RR_AS_F_26 = 0;
    env_RR_AS_F_18 = 0;
    env_RR_AS_M_46 = 0;
    env_RR_AS_M_26 = 0;
    env_RR_AS_M_18 = 0;
    env_BAR_BETA0 = 0;
    env_BAR_BETAA = 0;
    env_BAR_BETAS = 0;
    env_BAR_BETAAS = 0;
    env_BAR_SIZE = 0;
    env_SCHOOL_SIZE = 0;
    env_BAR_A = 0;
    env_BAR_V = 0;
    env_SCHOOL_A = 0;
    env_SCHOOL_V = 0;
    env_SEED = 0;
    env_HOUSEHOLD_EXP = 0;
    env_CHURCH_EXP = 0;
    env_TRANSPORT_EXP = 0;
    env_CLINIC_EXP = 0;
    env_WORKPLACE_EXP = 0;
    env_BAR_EXP = 0;
    env_SCHOOL_EXP = 0;
    env_PROB = 0;
    env_BAR_M_PROB1 = 0;
    env_BAR_M_PROB2 = 0;
    env_BAR_M_PROB3 = 0;
    env_BAR_M_PROB4 = 0;
    env_BAR_M_PROB5 = 0;
    env_BAR_M_PROB7 = 0;
    env_BAR_F_PROB1 = 0;
    env_BAR_F_PROB2 = 0;
    env_BAR_F_PROB3 = 0;
    env_BAR_F_PROB4 = 0;
    env_BAR_F_PROB5 = 0;
    env_BAR_F_PROB7 = 0;
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "Person") == 0)
				{
					if (*h_xmachine_memory_Person_count > xmachine_memory_Person_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Person exceeded whilst reading data\n", xmachine_memory_Person_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Persons->id[*h_xmachine_memory_Person_count] = Person_id;
					h_Persons->step[*h_xmachine_memory_Person_count] = Person_step;
					h_Persons->householdtime[*h_xmachine_memory_Person_count] = Person_householdtime;
					h_Persons->churchtime[*h_xmachine_memory_Person_count] = Person_churchtime;
					h_Persons->transporttime[*h_xmachine_memory_Person_count] = Person_transporttime;
					h_Persons->clinictime[*h_xmachine_memory_Person_count] = Person_clinictime;
					h_Persons->workplacetime[*h_xmachine_memory_Person_count] = Person_workplacetime;
					h_Persons->bartime[*h_xmachine_memory_Person_count] = Person_bartime;
					h_Persons->outsidetime[*h_xmachine_memory_Person_count] = Person_outsidetime;
					h_Persons->age[*h_xmachine_memory_Person_count] = Person_age;
					h_Persons->gender[*h_xmachine_memory_Person_count] = Person_gender;
					h_Persons->householdsize[*h_xmachine_memory_Person_count] = Person_householdsize;
					h_Persons->churchfreq[*h_xmachine_memory_Person_count] = Person_churchfreq;
					h_Persons->churchdur[*h_xmachine_memory_Person_count] = Person_churchdur;
					h_Persons->transportdur[*h_xmachine_memory_Person_count] = Person_transportdur;
					h_Persons->transportday1[*h_xmachine_memory_Person_count] = Person_transportday1;
					h_Persons->transportday2[*h_xmachine_memory_Person_count] = Person_transportday2;
					h_Persons->household[*h_xmachine_memory_Person_count] = Person_household;
					h_Persons->church[*h_xmachine_memory_Person_count] = Person_church;
					h_Persons->transport[*h_xmachine_memory_Person_count] = Person_transport;
					h_Persons->workplace[*h_xmachine_memory_Person_count] = Person_workplace;
					h_Persons->school[*h_xmachine_memory_Person_count] = Person_school;
					h_Persons->busy[*h_xmachine_memory_Person_count] = Person_busy;
					h_Persons->startstep[*h_xmachine_memory_Person_count] = Person_startstep;
					h_Persons->location[*h_xmachine_memory_Person_count] = Person_location;
					h_Persons->locationid[*h_xmachine_memory_Person_count] = Person_locationid;
					h_Persons->hiv[*h_xmachine_memory_Person_count] = Person_hiv;
					h_Persons->art[*h_xmachine_memory_Person_count] = Person_art;
					h_Persons->activetb[*h_xmachine_memory_Person_count] = Person_activetb;
					h_Persons->artday[*h_xmachine_memory_Person_count] = Person_artday;
					h_Persons->p[*h_xmachine_memory_Person_count] = Person_p;
					h_Persons->q[*h_xmachine_memory_Person_count] = Person_q;
					h_Persons->infections[*h_xmachine_memory_Person_count] = Person_infections;
					h_Persons->lastinfected[*h_xmachine_memory_Person_count] = Person_lastinfected;
					h_Persons->lastinfectedid[*h_xmachine_memory_Person_count] = Person_lastinfectedid;
					h_Persons->lambda[*h_xmachine_memory_Person_count] = Person_lambda;
					h_Persons->timevisiting[*h_xmachine_memory_Person_count] = Person_timevisiting;
					h_Persons->bargoing[*h_xmachine_memory_Person_count] = Person_bargoing;
					h_Persons->barday[*h_xmachine_memory_Person_count] = Person_barday;
					h_Persons->schooltime[*h_xmachine_memory_Person_count] = Person_schooltime;
					(*h_xmachine_memory_Person_count) ++;	
				}
				else if(strcmp(agentname, "TBAssignment") == 0)
				{
					if (*h_xmachine_memory_TBAssignment_count > xmachine_memory_TBAssignment_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent TBAssignment exceeded whilst reading data\n", xmachine_memory_TBAssignment_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_TBAssignments->id[*h_xmachine_memory_TBAssignment_count] = TBAssignment_id;
					(*h_xmachine_memory_TBAssignment_count) ++;	
				}
				else if(strcmp(agentname, "Household") == 0)
				{
					if (*h_xmachine_memory_Household_count > xmachine_memory_Household_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Household exceeded whilst reading data\n", xmachine_memory_Household_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Households->id[*h_xmachine_memory_Household_count] = Household_id;
					h_Households->lambda[*h_xmachine_memory_Household_count] = Household_lambda;
					h_Households->active[*h_xmachine_memory_Household_count] = Household_active;
					(*h_xmachine_memory_Household_count) ++;	
				}
				else if(strcmp(agentname, "HouseholdMembership") == 0)
				{
					if (*h_xmachine_memory_HouseholdMembership_count > xmachine_memory_HouseholdMembership_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent HouseholdMembership exceeded whilst reading data\n", xmachine_memory_HouseholdMembership_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_HouseholdMemberships->household_id[*h_xmachine_memory_HouseholdMembership_count] = HouseholdMembership_household_id;
					h_HouseholdMemberships->person_id[*h_xmachine_memory_HouseholdMembership_count] = HouseholdMembership_person_id;
					h_HouseholdMemberships->household_size[*h_xmachine_memory_HouseholdMembership_count] = HouseholdMembership_household_size;
					h_HouseholdMemberships->churchgoing[*h_xmachine_memory_HouseholdMembership_count] = HouseholdMembership_churchgoing;
					h_HouseholdMemberships->churchfreq[*h_xmachine_memory_HouseholdMembership_count] = HouseholdMembership_churchfreq;
					(*h_xmachine_memory_HouseholdMembership_count) ++;	
				}
				else if(strcmp(agentname, "Church") == 0)
				{
					if (*h_xmachine_memory_Church_count > xmachine_memory_Church_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Church exceeded whilst reading data\n", xmachine_memory_Church_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Churchs->id[*h_xmachine_memory_Church_count] = Church_id;
					h_Churchs->size[*h_xmachine_memory_Church_count] = Church_size;
					h_Churchs->lambda[*h_xmachine_memory_Church_count] = Church_lambda;
					h_Churchs->active[*h_xmachine_memory_Church_count] = Church_active;
					(*h_xmachine_memory_Church_count) ++;	
				}
				else if(strcmp(agentname, "ChurchMembership") == 0)
				{
					if (*h_xmachine_memory_ChurchMembership_count > xmachine_memory_ChurchMembership_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent ChurchMembership exceeded whilst reading data\n", xmachine_memory_ChurchMembership_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_ChurchMemberships->church_id[*h_xmachine_memory_ChurchMembership_count] = ChurchMembership_church_id;
					h_ChurchMemberships->household_id[*h_xmachine_memory_ChurchMembership_count] = ChurchMembership_household_id;
					h_ChurchMemberships->churchdur[*h_xmachine_memory_ChurchMembership_count] = ChurchMembership_churchdur;
					(*h_xmachine_memory_ChurchMembership_count) ++;	
				}
				else if(strcmp(agentname, "Transport") == 0)
				{
					if (*h_xmachine_memory_Transport_count > xmachine_memory_Transport_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Transport exceeded whilst reading data\n", xmachine_memory_Transport_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Transports->id[*h_xmachine_memory_Transport_count] = Transport_id;
					h_Transports->lambda[*h_xmachine_memory_Transport_count] = Transport_lambda;
					h_Transports->active[*h_xmachine_memory_Transport_count] = Transport_active;
					(*h_xmachine_memory_Transport_count) ++;	
				}
				else if(strcmp(agentname, "TransportMembership") == 0)
				{
					if (*h_xmachine_memory_TransportMembership_count > xmachine_memory_TransportMembership_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent TransportMembership exceeded whilst reading data\n", xmachine_memory_TransportMembership_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_TransportMemberships->person_id[*h_xmachine_memory_TransportMembership_count] = TransportMembership_person_id;
					h_TransportMemberships->transport_id[*h_xmachine_memory_TransportMembership_count] = TransportMembership_transport_id;
					h_TransportMemberships->duration[*h_xmachine_memory_TransportMembership_count] = TransportMembership_duration;
					(*h_xmachine_memory_TransportMembership_count) ++;	
				}
				else if(strcmp(agentname, "Clinic") == 0)
				{
					if (*h_xmachine_memory_Clinic_count > xmachine_memory_Clinic_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Clinic exceeded whilst reading data\n", xmachine_memory_Clinic_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Clinics->id[*h_xmachine_memory_Clinic_count] = Clinic_id;
					h_Clinics->lambda[*h_xmachine_memory_Clinic_count] = Clinic_lambda;
					(*h_xmachine_memory_Clinic_count) ++;	
				}
				else if(strcmp(agentname, "Workplace") == 0)
				{
					if (*h_xmachine_memory_Workplace_count > xmachine_memory_Workplace_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Workplace exceeded whilst reading data\n", xmachine_memory_Workplace_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Workplaces->id[*h_xmachine_memory_Workplace_count] = Workplace_id;
					h_Workplaces->lambda[*h_xmachine_memory_Workplace_count] = Workplace_lambda;
					(*h_xmachine_memory_Workplace_count) ++;	
				}
				else if(strcmp(agentname, "WorkplaceMembership") == 0)
				{
					if (*h_xmachine_memory_WorkplaceMembership_count > xmachine_memory_WorkplaceMembership_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent WorkplaceMembership exceeded whilst reading data\n", xmachine_memory_WorkplaceMembership_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_WorkplaceMemberships->person_id[*h_xmachine_memory_WorkplaceMembership_count] = WorkplaceMembership_person_id;
					h_WorkplaceMemberships->workplace_id[*h_xmachine_memory_WorkplaceMembership_count] = WorkplaceMembership_workplace_id;
					(*h_xmachine_memory_WorkplaceMembership_count) ++;	
				}
				else if(strcmp(agentname, "Bar") == 0)
				{
					if (*h_xmachine_memory_Bar_count > xmachine_memory_Bar_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Bar exceeded whilst reading data\n", xmachine_memory_Bar_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Bars->id[*h_xmachine_memory_Bar_count] = Bar_id;
					h_Bars->lambda[*h_xmachine_memory_Bar_count] = Bar_lambda;
					(*h_xmachine_memory_Bar_count) ++;	
				}
				else if(strcmp(agentname, "School") == 0)
				{
					if (*h_xmachine_memory_School_count > xmachine_memory_School_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent School exceeded whilst reading data\n", xmachine_memory_School_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Schools->id[*h_xmachine_memory_School_count] = School_id;
					h_Schools->lambda[*h_xmachine_memory_School_count] = School_lambda;
					(*h_xmachine_memory_School_count) ++;	
				}
				else if(strcmp(agentname, "SchoolMembership") == 0)
				{
					if (*h_xmachine_memory_SchoolMembership_count > xmachine_memory_SchoolMembership_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent SchoolMembership exceeded whilst reading data\n", xmachine_memory_SchoolMembership_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_SchoolMemberships->person_id[*h_xmachine_memory_SchoolMembership_count] = SchoolMembership_person_id;
					h_SchoolMemberships->school_id[*h_xmachine_memory_SchoolMembership_count] = SchoolMembership_school_id;
					(*h_xmachine_memory_SchoolMembership_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Person_id = 0;
                Person_step = 0;
                Person_householdtime = 0;
                Person_churchtime = 0;
                Person_transporttime = 0;
                Person_clinictime = 0;
                Person_workplacetime = 0;
                Person_bartime = 0;
                Person_outsidetime = 0;
                Person_age = 0;
                Person_gender = 0;
                Person_householdsize = 0;
                Person_churchfreq = 0;
                Person_churchdur = 0;
                Person_transportdur = 0;
                Person_transportday1 = 0;
                Person_transportday2 = 0;
                Person_household = 0;
                Person_church = 0;
                Person_transport = 0;
                Person_workplace = 0;
                Person_school = 0;
                Person_busy = 0;
                Person_startstep = 0;
                Person_location = 0;
                Person_locationid = 0;
                Person_hiv = 0;
                Person_art = 0;
                Person_activetb = 0;
                Person_artday = 0;
                Person_p = 0;
                Person_q = 0;
                Person_infections = 0;
                Person_lastinfected = 0;
                Person_lastinfectedid = 0;
                Person_lambda = 0;
                Person_timevisiting = 0;
                Person_bargoing = 0;
                Person_barday = 0;
                Person_schooltime = 0;
                TBAssignment_id = 0;
                Household_id = 0;
                Household_lambda = 0;
                Household_active = 0;
                HouseholdMembership_household_id = 0;
                HouseholdMembership_person_id = 0;
                HouseholdMembership_household_size = 0;
                HouseholdMembership_churchgoing = 0;
                HouseholdMembership_churchfreq = 0;
                Church_id = 0;
                Church_size = 0;
                Church_lambda = 0;
                Church_active = 0;
                ChurchMembership_church_id = 0;
                ChurchMembership_household_id = 0;
                ChurchMembership_churchdur = 0;
                Transport_id = 0;
                Transport_lambda = 0;
                Transport_active = 0;
                TransportMembership_person_id = 0;
                TransportMembership_transport_id = 0;
                TransportMembership_duration = 0;
                Clinic_id = 0;
                Clinic_lambda = 0;
                Workplace_id = 0;
                Workplace_lambda = 0;
                WorkplaceMembership_person_id = 0;
                WorkplaceMembership_workplace_id = 0;
                Bar_id = 0;
                Bar_lambda = 0;
                School_id = 0;
                School_lambda = 0;
                SchoolMembership_person_id = 0;
                SchoolMembership_school_id = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Person_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Person_id = 0;
			if(strcmp(buffer, "step") == 0) in_Person_step = 1;
			if(strcmp(buffer, "/step") == 0) in_Person_step = 0;
			if(strcmp(buffer, "householdtime") == 0) in_Person_householdtime = 1;
			if(strcmp(buffer, "/householdtime") == 0) in_Person_householdtime = 0;
			if(strcmp(buffer, "churchtime") == 0) in_Person_churchtime = 1;
			if(strcmp(buffer, "/churchtime") == 0) in_Person_churchtime = 0;
			if(strcmp(buffer, "transporttime") == 0) in_Person_transporttime = 1;
			if(strcmp(buffer, "/transporttime") == 0) in_Person_transporttime = 0;
			if(strcmp(buffer, "clinictime") == 0) in_Person_clinictime = 1;
			if(strcmp(buffer, "/clinictime") == 0) in_Person_clinictime = 0;
			if(strcmp(buffer, "workplacetime") == 0) in_Person_workplacetime = 1;
			if(strcmp(buffer, "/workplacetime") == 0) in_Person_workplacetime = 0;
			if(strcmp(buffer, "bartime") == 0) in_Person_bartime = 1;
			if(strcmp(buffer, "/bartime") == 0) in_Person_bartime = 0;
			if(strcmp(buffer, "outsidetime") == 0) in_Person_outsidetime = 1;
			if(strcmp(buffer, "/outsidetime") == 0) in_Person_outsidetime = 0;
			if(strcmp(buffer, "age") == 0) in_Person_age = 1;
			if(strcmp(buffer, "/age") == 0) in_Person_age = 0;
			if(strcmp(buffer, "gender") == 0) in_Person_gender = 1;
			if(strcmp(buffer, "/gender") == 0) in_Person_gender = 0;
			if(strcmp(buffer, "householdsize") == 0) in_Person_householdsize = 1;
			if(strcmp(buffer, "/householdsize") == 0) in_Person_householdsize = 0;
			if(strcmp(buffer, "churchfreq") == 0) in_Person_churchfreq = 1;
			if(strcmp(buffer, "/churchfreq") == 0) in_Person_churchfreq = 0;
			if(strcmp(buffer, "churchdur") == 0) in_Person_churchdur = 1;
			if(strcmp(buffer, "/churchdur") == 0) in_Person_churchdur = 0;
			if(strcmp(buffer, "transportdur") == 0) in_Person_transportdur = 1;
			if(strcmp(buffer, "/transportdur") == 0) in_Person_transportdur = 0;
			if(strcmp(buffer, "transportday1") == 0) in_Person_transportday1 = 1;
			if(strcmp(buffer, "/transportday1") == 0) in_Person_transportday1 = 0;
			if(strcmp(buffer, "transportday2") == 0) in_Person_transportday2 = 1;
			if(strcmp(buffer, "/transportday2") == 0) in_Person_transportday2 = 0;
			if(strcmp(buffer, "household") == 0) in_Person_household = 1;
			if(strcmp(buffer, "/household") == 0) in_Person_household = 0;
			if(strcmp(buffer, "church") == 0) in_Person_church = 1;
			if(strcmp(buffer, "/church") == 0) in_Person_church = 0;
			if(strcmp(buffer, "transport") == 0) in_Person_transport = 1;
			if(strcmp(buffer, "/transport") == 0) in_Person_transport = 0;
			if(strcmp(buffer, "workplace") == 0) in_Person_workplace = 1;
			if(strcmp(buffer, "/workplace") == 0) in_Person_workplace = 0;
			if(strcmp(buffer, "school") == 0) in_Person_school = 1;
			if(strcmp(buffer, "/school") == 0) in_Person_school = 0;
			if(strcmp(buffer, "busy") == 0) in_Person_busy = 1;
			if(strcmp(buffer, "/busy") == 0) in_Person_busy = 0;
			if(strcmp(buffer, "startstep") == 0) in_Person_startstep = 1;
			if(strcmp(buffer, "/startstep") == 0) in_Person_startstep = 0;
			if(strcmp(buffer, "location") == 0) in_Person_location = 1;
			if(strcmp(buffer, "/location") == 0) in_Person_location = 0;
			if(strcmp(buffer, "locationid") == 0) in_Person_locationid = 1;
			if(strcmp(buffer, "/locationid") == 0) in_Person_locationid = 0;
			if(strcmp(buffer, "hiv") == 0) in_Person_hiv = 1;
			if(strcmp(buffer, "/hiv") == 0) in_Person_hiv = 0;
			if(strcmp(buffer, "art") == 0) in_Person_art = 1;
			if(strcmp(buffer, "/art") == 0) in_Person_art = 0;
			if(strcmp(buffer, "activetb") == 0) in_Person_activetb = 1;
			if(strcmp(buffer, "/activetb") == 0) in_Person_activetb = 0;
			if(strcmp(buffer, "artday") == 0) in_Person_artday = 1;
			if(strcmp(buffer, "/artday") == 0) in_Person_artday = 0;
			if(strcmp(buffer, "p") == 0) in_Person_p = 1;
			if(strcmp(buffer, "/p") == 0) in_Person_p = 0;
			if(strcmp(buffer, "q") == 0) in_Person_q = 1;
			if(strcmp(buffer, "/q") == 0) in_Person_q = 0;
			if(strcmp(buffer, "infections") == 0) in_Person_infections = 1;
			if(strcmp(buffer, "/infections") == 0) in_Person_infections = 0;
			if(strcmp(buffer, "lastinfected") == 0) in_Person_lastinfected = 1;
			if(strcmp(buffer, "/lastinfected") == 0) in_Person_lastinfected = 0;
			if(strcmp(buffer, "lastinfectedid") == 0) in_Person_lastinfectedid = 1;
			if(strcmp(buffer, "/lastinfectedid") == 0) in_Person_lastinfectedid = 0;
			if(strcmp(buffer, "lambda") == 0) in_Person_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Person_lambda = 0;
			if(strcmp(buffer, "timevisiting") == 0) in_Person_timevisiting = 1;
			if(strcmp(buffer, "/timevisiting") == 0) in_Person_timevisiting = 0;
			if(strcmp(buffer, "bargoing") == 0) in_Person_bargoing = 1;
			if(strcmp(buffer, "/bargoing") == 0) in_Person_bargoing = 0;
			if(strcmp(buffer, "barday") == 0) in_Person_barday = 1;
			if(strcmp(buffer, "/barday") == 0) in_Person_barday = 0;
			if(strcmp(buffer, "schooltime") == 0) in_Person_schooltime = 1;
			if(strcmp(buffer, "/schooltime") == 0) in_Person_schooltime = 0;
			if(strcmp(buffer, "id") == 0) in_TBAssignment_id = 1;
			if(strcmp(buffer, "/id") == 0) in_TBAssignment_id = 0;
			if(strcmp(buffer, "id") == 0) in_Household_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Household_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_Household_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Household_lambda = 0;
			if(strcmp(buffer, "active") == 0) in_Household_active = 1;
			if(strcmp(buffer, "/active") == 0) in_Household_active = 0;
			if(strcmp(buffer, "household_id") == 0) in_HouseholdMembership_household_id = 1;
			if(strcmp(buffer, "/household_id") == 0) in_HouseholdMembership_household_id = 0;
			if(strcmp(buffer, "person_id") == 0) in_HouseholdMembership_person_id = 1;
			if(strcmp(buffer, "/person_id") == 0) in_HouseholdMembership_person_id = 0;
			if(strcmp(buffer, "household_size") == 0) in_HouseholdMembership_household_size = 1;
			if(strcmp(buffer, "/household_size") == 0) in_HouseholdMembership_household_size = 0;
			if(strcmp(buffer, "churchgoing") == 0) in_HouseholdMembership_churchgoing = 1;
			if(strcmp(buffer, "/churchgoing") == 0) in_HouseholdMembership_churchgoing = 0;
			if(strcmp(buffer, "churchfreq") == 0) in_HouseholdMembership_churchfreq = 1;
			if(strcmp(buffer, "/churchfreq") == 0) in_HouseholdMembership_churchfreq = 0;
			if(strcmp(buffer, "id") == 0) in_Church_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Church_id = 0;
			if(strcmp(buffer, "size") == 0) in_Church_size = 1;
			if(strcmp(buffer, "/size") == 0) in_Church_size = 0;
			if(strcmp(buffer, "lambda") == 0) in_Church_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Church_lambda = 0;
			if(strcmp(buffer, "active") == 0) in_Church_active = 1;
			if(strcmp(buffer, "/active") == 0) in_Church_active = 0;
			if(strcmp(buffer, "church_id") == 0) in_ChurchMembership_church_id = 1;
			if(strcmp(buffer, "/church_id") == 0) in_ChurchMembership_church_id = 0;
			if(strcmp(buffer, "household_id") == 0) in_ChurchMembership_household_id = 1;
			if(strcmp(buffer, "/household_id") == 0) in_ChurchMembership_household_id = 0;
			if(strcmp(buffer, "churchdur") == 0) in_ChurchMembership_churchdur = 1;
			if(strcmp(buffer, "/churchdur") == 0) in_ChurchMembership_churchdur = 0;
			if(strcmp(buffer, "id") == 0) in_Transport_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Transport_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_Transport_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Transport_lambda = 0;
			if(strcmp(buffer, "active") == 0) in_Transport_active = 1;
			if(strcmp(buffer, "/active") == 0) in_Transport_active = 0;
			if(strcmp(buffer, "person_id") == 0) in_TransportMembership_person_id = 1;
			if(strcmp(buffer, "/person_id") == 0) in_TransportMembership_person_id = 0;
			if(strcmp(buffer, "transport_id") == 0) in_TransportMembership_transport_id = 1;
			if(strcmp(buffer, "/transport_id") == 0) in_TransportMembership_transport_id = 0;
			if(strcmp(buffer, "duration") == 0) in_TransportMembership_duration = 1;
			if(strcmp(buffer, "/duration") == 0) in_TransportMembership_duration = 0;
			if(strcmp(buffer, "id") == 0) in_Clinic_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Clinic_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_Clinic_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Clinic_lambda = 0;
			if(strcmp(buffer, "id") == 0) in_Workplace_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Workplace_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_Workplace_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Workplace_lambda = 0;
			if(strcmp(buffer, "person_id") == 0) in_WorkplaceMembership_person_id = 1;
			if(strcmp(buffer, "/person_id") == 0) in_WorkplaceMembership_person_id = 0;
			if(strcmp(buffer, "workplace_id") == 0) in_WorkplaceMembership_workplace_id = 1;
			if(strcmp(buffer, "/workplace_id") == 0) in_WorkplaceMembership_workplace_id = 0;
			if(strcmp(buffer, "id") == 0) in_Bar_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Bar_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_Bar_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_Bar_lambda = 0;
			if(strcmp(buffer, "id") == 0) in_School_id = 1;
			if(strcmp(buffer, "/id") == 0) in_School_id = 0;
			if(strcmp(buffer, "lambda") == 0) in_School_lambda = 1;
			if(strcmp(buffer, "/lambda") == 0) in_School_lambda = 0;
			if(strcmp(buffer, "person_id") == 0) in_SchoolMembership_person_id = 1;
			if(strcmp(buffer, "/person_id") == 0) in_SchoolMembership_person_id = 0;
			if(strcmp(buffer, "school_id") == 0) in_SchoolMembership_school_id = 1;
			if(strcmp(buffer, "/school_id") == 0) in_SchoolMembership_school_id = 0;
			
            /* environment variables */
            if(strcmp(buffer, "TIME_STEP") == 0) in_env_TIME_STEP = 1;
            if(strcmp(buffer, "/TIME_STEP") == 0) in_env_TIME_STEP = 0;
			if(strcmp(buffer, "MAX_AGE") == 0) in_env_MAX_AGE = 1;
            if(strcmp(buffer, "/MAX_AGE") == 0) in_env_MAX_AGE = 0;
			if(strcmp(buffer, "STARTING_POPULATION") == 0) in_env_STARTING_POPULATION = 1;
            if(strcmp(buffer, "/STARTING_POPULATION") == 0) in_env_STARTING_POPULATION = 0;
			if(strcmp(buffer, "CHURCH_BETA0") == 0) in_env_CHURCH_BETA0 = 1;
            if(strcmp(buffer, "/CHURCH_BETA0") == 0) in_env_CHURCH_BETA0 = 0;
			if(strcmp(buffer, "CHURCH_BETA1") == 0) in_env_CHURCH_BETA1 = 1;
            if(strcmp(buffer, "/CHURCH_BETA1") == 0) in_env_CHURCH_BETA1 = 0;
			if(strcmp(buffer, "CHURCH_K1") == 0) in_env_CHURCH_K1 = 1;
            if(strcmp(buffer, "/CHURCH_K1") == 0) in_env_CHURCH_K1 = 0;
			if(strcmp(buffer, "CHURCH_K2") == 0) in_env_CHURCH_K2 = 1;
            if(strcmp(buffer, "/CHURCH_K2") == 0) in_env_CHURCH_K2 = 0;
			if(strcmp(buffer, "CHURCH_K3") == 0) in_env_CHURCH_K3 = 1;
            if(strcmp(buffer, "/CHURCH_K3") == 0) in_env_CHURCH_K3 = 0;
			if(strcmp(buffer, "CHURCH_P1") == 0) in_env_CHURCH_P1 = 1;
            if(strcmp(buffer, "/CHURCH_P1") == 0) in_env_CHURCH_P1 = 0;
			if(strcmp(buffer, "CHURCH_P2") == 0) in_env_CHURCH_P2 = 1;
            if(strcmp(buffer, "/CHURCH_P2") == 0) in_env_CHURCH_P2 = 0;
			if(strcmp(buffer, "CHURCH_PROB0") == 0) in_env_CHURCH_PROB0 = 1;
            if(strcmp(buffer, "/CHURCH_PROB0") == 0) in_env_CHURCH_PROB0 = 0;
			if(strcmp(buffer, "CHURCH_PROB1") == 0) in_env_CHURCH_PROB1 = 1;
            if(strcmp(buffer, "/CHURCH_PROB1") == 0) in_env_CHURCH_PROB1 = 0;
			if(strcmp(buffer, "CHURCH_PROB2") == 0) in_env_CHURCH_PROB2 = 1;
            if(strcmp(buffer, "/CHURCH_PROB2") == 0) in_env_CHURCH_PROB2 = 0;
			if(strcmp(buffer, "CHURCH_PROB3") == 0) in_env_CHURCH_PROB3 = 1;
            if(strcmp(buffer, "/CHURCH_PROB3") == 0) in_env_CHURCH_PROB3 = 0;
			if(strcmp(buffer, "CHURCH_PROB4") == 0) in_env_CHURCH_PROB4 = 1;
            if(strcmp(buffer, "/CHURCH_PROB4") == 0) in_env_CHURCH_PROB4 = 0;
			if(strcmp(buffer, "CHURCH_PROB5") == 0) in_env_CHURCH_PROB5 = 1;
            if(strcmp(buffer, "/CHURCH_PROB5") == 0) in_env_CHURCH_PROB5 = 0;
			if(strcmp(buffer, "CHURCH_PROB6") == 0) in_env_CHURCH_PROB6 = 1;
            if(strcmp(buffer, "/CHURCH_PROB6") == 0) in_env_CHURCH_PROB6 = 0;
			if(strcmp(buffer, "CHURCH_DURATION") == 0) in_env_CHURCH_DURATION = 1;
            if(strcmp(buffer, "/CHURCH_DURATION") == 0) in_env_CHURCH_DURATION = 0;
			if(strcmp(buffer, "TRANSPORT_BETA0") == 0) in_env_TRANSPORT_BETA0 = 1;
            if(strcmp(buffer, "/TRANSPORT_BETA0") == 0) in_env_TRANSPORT_BETA0 = 0;
			if(strcmp(buffer, "TRANSPORT_BETA1") == 0) in_env_TRANSPORT_BETA1 = 1;
            if(strcmp(buffer, "/TRANSPORT_BETA1") == 0) in_env_TRANSPORT_BETA1 = 0;
			if(strcmp(buffer, "TRANSPORT_FREQ0") == 0) in_env_TRANSPORT_FREQ0 = 1;
            if(strcmp(buffer, "/TRANSPORT_FREQ0") == 0) in_env_TRANSPORT_FREQ0 = 0;
			if(strcmp(buffer, "TRANSPORT_FREQ2") == 0) in_env_TRANSPORT_FREQ2 = 1;
            if(strcmp(buffer, "/TRANSPORT_FREQ2") == 0) in_env_TRANSPORT_FREQ2 = 0;
			if(strcmp(buffer, "TRANSPORT_DUR20") == 0) in_env_TRANSPORT_DUR20 = 1;
            if(strcmp(buffer, "/TRANSPORT_DUR20") == 0) in_env_TRANSPORT_DUR20 = 0;
			if(strcmp(buffer, "TRANSPORT_DUR45") == 0) in_env_TRANSPORT_DUR45 = 1;
            if(strcmp(buffer, "/TRANSPORT_DUR45") == 0) in_env_TRANSPORT_DUR45 = 0;
			if(strcmp(buffer, "TRANSPORT_SIZE") == 0) in_env_TRANSPORT_SIZE = 1;
            if(strcmp(buffer, "/TRANSPORT_SIZE") == 0) in_env_TRANSPORT_SIZE = 0;
			if(strcmp(buffer, "HIV_PREVALENCE") == 0) in_env_HIV_PREVALENCE = 1;
            if(strcmp(buffer, "/HIV_PREVALENCE") == 0) in_env_HIV_PREVALENCE = 0;
			if(strcmp(buffer, "ART_COVERAGE") == 0) in_env_ART_COVERAGE = 1;
            if(strcmp(buffer, "/ART_COVERAGE") == 0) in_env_ART_COVERAGE = 0;
			if(strcmp(buffer, "RR_HIV") == 0) in_env_RR_HIV = 1;
            if(strcmp(buffer, "/RR_HIV") == 0) in_env_RR_HIV = 0;
			if(strcmp(buffer, "RR_ART") == 0) in_env_RR_ART = 1;
            if(strcmp(buffer, "/RR_ART") == 0) in_env_RR_ART = 0;
			if(strcmp(buffer, "TB_PREVALENCE") == 0) in_env_TB_PREVALENCE = 1;
            if(strcmp(buffer, "/TB_PREVALENCE") == 0) in_env_TB_PREVALENCE = 0;
			if(strcmp(buffer, "DEFAULT_P") == 0) in_env_DEFAULT_P = 1;
            if(strcmp(buffer, "/DEFAULT_P") == 0) in_env_DEFAULT_P = 0;
			if(strcmp(buffer, "DEFAULT_Q") == 0) in_env_DEFAULT_Q = 1;
            if(strcmp(buffer, "/DEFAULT_Q") == 0) in_env_DEFAULT_Q = 0;
			if(strcmp(buffer, "TRANSPORT_A") == 0) in_env_TRANSPORT_A = 1;
            if(strcmp(buffer, "/TRANSPORT_A") == 0) in_env_TRANSPORT_A = 0;
			if(strcmp(buffer, "CHURCH_A") == 0) in_env_CHURCH_A = 1;
            if(strcmp(buffer, "/CHURCH_A") == 0) in_env_CHURCH_A = 0;
			if(strcmp(buffer, "CLINIC_A") == 0) in_env_CLINIC_A = 1;
            if(strcmp(buffer, "/CLINIC_A") == 0) in_env_CLINIC_A = 0;
			if(strcmp(buffer, "HOUSEHOLD_A") == 0) in_env_HOUSEHOLD_A = 1;
            if(strcmp(buffer, "/HOUSEHOLD_A") == 0) in_env_HOUSEHOLD_A = 0;
			if(strcmp(buffer, "TRANSPORT_V") == 0) in_env_TRANSPORT_V = 1;
            if(strcmp(buffer, "/TRANSPORT_V") == 0) in_env_TRANSPORT_V = 0;
			if(strcmp(buffer, "HOUSEHOLD_V") == 0) in_env_HOUSEHOLD_V = 1;
            if(strcmp(buffer, "/HOUSEHOLD_V") == 0) in_env_HOUSEHOLD_V = 0;
			if(strcmp(buffer, "CLINIC_V") == 0) in_env_CLINIC_V = 1;
            if(strcmp(buffer, "/CLINIC_V") == 0) in_env_CLINIC_V = 0;
			if(strcmp(buffer, "CHURCH_V_MULTIPLIER") == 0) in_env_CHURCH_V_MULTIPLIER = 1;
            if(strcmp(buffer, "/CHURCH_V_MULTIPLIER") == 0) in_env_CHURCH_V_MULTIPLIER = 0;
			if(strcmp(buffer, "WORKPLACE_BETA0") == 0) in_env_WORKPLACE_BETA0 = 1;
            if(strcmp(buffer, "/WORKPLACE_BETA0") == 0) in_env_WORKPLACE_BETA0 = 0;
			if(strcmp(buffer, "WORKPLACE_BETAA") == 0) in_env_WORKPLACE_BETAA = 1;
            if(strcmp(buffer, "/WORKPLACE_BETAA") == 0) in_env_WORKPLACE_BETAA = 0;
			if(strcmp(buffer, "WORKPLACE_BETAS") == 0) in_env_WORKPLACE_BETAS = 1;
            if(strcmp(buffer, "/WORKPLACE_BETAS") == 0) in_env_WORKPLACE_BETAS = 0;
			if(strcmp(buffer, "WORKPLACE_BETAAS") == 0) in_env_WORKPLACE_BETAAS = 1;
            if(strcmp(buffer, "/WORKPLACE_BETAAS") == 0) in_env_WORKPLACE_BETAAS = 0;
			if(strcmp(buffer, "WORKPLACE_A") == 0) in_env_WORKPLACE_A = 1;
            if(strcmp(buffer, "/WORKPLACE_A") == 0) in_env_WORKPLACE_A = 0;
			if(strcmp(buffer, "WORKPLACE_DUR") == 0) in_env_WORKPLACE_DUR = 1;
            if(strcmp(buffer, "/WORKPLACE_DUR") == 0) in_env_WORKPLACE_DUR = 0;
			if(strcmp(buffer, "WORKPLACE_SIZE") == 0) in_env_WORKPLACE_SIZE = 1;
            if(strcmp(buffer, "/WORKPLACE_SIZE") == 0) in_env_WORKPLACE_SIZE = 0;
			if(strcmp(buffer, "WORKPLACE_V") == 0) in_env_WORKPLACE_V = 1;
            if(strcmp(buffer, "/WORKPLACE_V") == 0) in_env_WORKPLACE_V = 0;
			if(strcmp(buffer, "HOUSEHOLDS") == 0) in_env_HOUSEHOLDS = 1;
            if(strcmp(buffer, "/HOUSEHOLDS") == 0) in_env_HOUSEHOLDS = 0;
			if(strcmp(buffer, "BARS") == 0) in_env_BARS = 1;
            if(strcmp(buffer, "/BARS") == 0) in_env_BARS = 0;
			if(strcmp(buffer, "RR_AS_F_46") == 0) in_env_RR_AS_F_46 = 1;
            if(strcmp(buffer, "/RR_AS_F_46") == 0) in_env_RR_AS_F_46 = 0;
			if(strcmp(buffer, "RR_AS_F_26") == 0) in_env_RR_AS_F_26 = 1;
            if(strcmp(buffer, "/RR_AS_F_26") == 0) in_env_RR_AS_F_26 = 0;
			if(strcmp(buffer, "RR_AS_F_18") == 0) in_env_RR_AS_F_18 = 1;
            if(strcmp(buffer, "/RR_AS_F_18") == 0) in_env_RR_AS_F_18 = 0;
			if(strcmp(buffer, "RR_AS_M_46") == 0) in_env_RR_AS_M_46 = 1;
            if(strcmp(buffer, "/RR_AS_M_46") == 0) in_env_RR_AS_M_46 = 0;
			if(strcmp(buffer, "RR_AS_M_26") == 0) in_env_RR_AS_M_26 = 1;
            if(strcmp(buffer, "/RR_AS_M_26") == 0) in_env_RR_AS_M_26 = 0;
			if(strcmp(buffer, "RR_AS_M_18") == 0) in_env_RR_AS_M_18 = 1;
            if(strcmp(buffer, "/RR_AS_M_18") == 0) in_env_RR_AS_M_18 = 0;
			if(strcmp(buffer, "BAR_BETA0") == 0) in_env_BAR_BETA0 = 1;
            if(strcmp(buffer, "/BAR_BETA0") == 0) in_env_BAR_BETA0 = 0;
			if(strcmp(buffer, "BAR_BETAA") == 0) in_env_BAR_BETAA = 1;
            if(strcmp(buffer, "/BAR_BETAA") == 0) in_env_BAR_BETAA = 0;
			if(strcmp(buffer, "BAR_BETAS") == 0) in_env_BAR_BETAS = 1;
            if(strcmp(buffer, "/BAR_BETAS") == 0) in_env_BAR_BETAS = 0;
			if(strcmp(buffer, "BAR_BETAAS") == 0) in_env_BAR_BETAAS = 1;
            if(strcmp(buffer, "/BAR_BETAAS") == 0) in_env_BAR_BETAAS = 0;
			if(strcmp(buffer, "BAR_SIZE") == 0) in_env_BAR_SIZE = 1;
            if(strcmp(buffer, "/BAR_SIZE") == 0) in_env_BAR_SIZE = 0;
			if(strcmp(buffer, "SCHOOL_SIZE") == 0) in_env_SCHOOL_SIZE = 1;
            if(strcmp(buffer, "/SCHOOL_SIZE") == 0) in_env_SCHOOL_SIZE = 0;
			if(strcmp(buffer, "BAR_A") == 0) in_env_BAR_A = 1;
            if(strcmp(buffer, "/BAR_A") == 0) in_env_BAR_A = 0;
			if(strcmp(buffer, "BAR_V") == 0) in_env_BAR_V = 1;
            if(strcmp(buffer, "/BAR_V") == 0) in_env_BAR_V = 0;
			if(strcmp(buffer, "SCHOOL_A") == 0) in_env_SCHOOL_A = 1;
            if(strcmp(buffer, "/SCHOOL_A") == 0) in_env_SCHOOL_A = 0;
			if(strcmp(buffer, "SCHOOL_V") == 0) in_env_SCHOOL_V = 1;
            if(strcmp(buffer, "/SCHOOL_V") == 0) in_env_SCHOOL_V = 0;
			if(strcmp(buffer, "SEED") == 0) in_env_SEED = 1;
            if(strcmp(buffer, "/SEED") == 0) in_env_SEED = 0;
			if(strcmp(buffer, "HOUSEHOLD_EXP") == 0) in_env_HOUSEHOLD_EXP = 1;
            if(strcmp(buffer, "/HOUSEHOLD_EXP") == 0) in_env_HOUSEHOLD_EXP = 0;
			if(strcmp(buffer, "CHURCH_EXP") == 0) in_env_CHURCH_EXP = 1;
            if(strcmp(buffer, "/CHURCH_EXP") == 0) in_env_CHURCH_EXP = 0;
			if(strcmp(buffer, "TRANSPORT_EXP") == 0) in_env_TRANSPORT_EXP = 1;
            if(strcmp(buffer, "/TRANSPORT_EXP") == 0) in_env_TRANSPORT_EXP = 0;
			if(strcmp(buffer, "CLINIC_EXP") == 0) in_env_CLINIC_EXP = 1;
            if(strcmp(buffer, "/CLINIC_EXP") == 0) in_env_CLINIC_EXP = 0;
			if(strcmp(buffer, "WORKPLACE_EXP") == 0) in_env_WORKPLACE_EXP = 1;
            if(strcmp(buffer, "/WORKPLACE_EXP") == 0) in_env_WORKPLACE_EXP = 0;
			if(strcmp(buffer, "BAR_EXP") == 0) in_env_BAR_EXP = 1;
            if(strcmp(buffer, "/BAR_EXP") == 0) in_env_BAR_EXP = 0;
			if(strcmp(buffer, "SCHOOL_EXP") == 0) in_env_SCHOOL_EXP = 1;
            if(strcmp(buffer, "/SCHOOL_EXP") == 0) in_env_SCHOOL_EXP = 0;
			if(strcmp(buffer, "PROB") == 0) in_env_PROB = 1;
            if(strcmp(buffer, "/PROB") == 0) in_env_PROB = 0;
			if(strcmp(buffer, "BAR_M_PROB1") == 0) in_env_BAR_M_PROB1 = 1;
            if(strcmp(buffer, "/BAR_M_PROB1") == 0) in_env_BAR_M_PROB1 = 0;
			if(strcmp(buffer, "BAR_M_PROB2") == 0) in_env_BAR_M_PROB2 = 1;
            if(strcmp(buffer, "/BAR_M_PROB2") == 0) in_env_BAR_M_PROB2 = 0;
			if(strcmp(buffer, "BAR_M_PROB3") == 0) in_env_BAR_M_PROB3 = 1;
            if(strcmp(buffer, "/BAR_M_PROB3") == 0) in_env_BAR_M_PROB3 = 0;
			if(strcmp(buffer, "BAR_M_PROB4") == 0) in_env_BAR_M_PROB4 = 1;
            if(strcmp(buffer, "/BAR_M_PROB4") == 0) in_env_BAR_M_PROB4 = 0;
			if(strcmp(buffer, "BAR_M_PROB5") == 0) in_env_BAR_M_PROB5 = 1;
            if(strcmp(buffer, "/BAR_M_PROB5") == 0) in_env_BAR_M_PROB5 = 0;
			if(strcmp(buffer, "BAR_M_PROB7") == 0) in_env_BAR_M_PROB7 = 1;
            if(strcmp(buffer, "/BAR_M_PROB7") == 0) in_env_BAR_M_PROB7 = 0;
			if(strcmp(buffer, "BAR_F_PROB1") == 0) in_env_BAR_F_PROB1 = 1;
            if(strcmp(buffer, "/BAR_F_PROB1") == 0) in_env_BAR_F_PROB1 = 0;
			if(strcmp(buffer, "BAR_F_PROB2") == 0) in_env_BAR_F_PROB2 = 1;
            if(strcmp(buffer, "/BAR_F_PROB2") == 0) in_env_BAR_F_PROB2 = 0;
			if(strcmp(buffer, "BAR_F_PROB3") == 0) in_env_BAR_F_PROB3 = 1;
            if(strcmp(buffer, "/BAR_F_PROB3") == 0) in_env_BAR_F_PROB3 = 0;
			if(strcmp(buffer, "BAR_F_PROB4") == 0) in_env_BAR_F_PROB4 = 1;
            if(strcmp(buffer, "/BAR_F_PROB4") == 0) in_env_BAR_F_PROB4 = 0;
			if(strcmp(buffer, "BAR_F_PROB5") == 0) in_env_BAR_F_PROB5 = 1;
            if(strcmp(buffer, "/BAR_F_PROB5") == 0) in_env_BAR_F_PROB5 = 0;
			if(strcmp(buffer, "BAR_F_PROB7") == 0) in_env_BAR_F_PROB7 = 1;
            if(strcmp(buffer, "/BAR_F_PROB7") == 0) in_env_BAR_F_PROB7 = 0;
			

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_Person_id){
                    Person_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_step){
                    Person_step = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_householdtime){
                    Person_householdtime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_churchtime){
                    Person_churchtime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_transporttime){
                    Person_transporttime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_clinictime){
                    Person_clinictime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_workplacetime){
                    Person_workplacetime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_bartime){
                    Person_bartime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_outsidetime){
                    Person_outsidetime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_age){
                    Person_age = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_gender){
                    Person_gender = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_householdsize){
                    Person_householdsize = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_churchfreq){
                    Person_churchfreq = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_churchdur){
                    Person_churchdur = (float) fgpu_atof(buffer); 
                }
				if(in_Person_transportdur){
                    Person_transportdur = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_transportday1){
                    Person_transportday1 = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_transportday2){
                    Person_transportday2 = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_household){
                    Person_household = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_church){
                    Person_church = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_transport){
                    Person_transport = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_workplace){
                    Person_workplace = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_school){
                    Person_school = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_busy){
                    Person_busy = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_startstep){
                    Person_startstep = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_location){
                    Person_location = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_locationid){
                    Person_locationid = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_hiv){
                    Person_hiv = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_art){
                    Person_art = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_activetb){
                    Person_activetb = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_artday){
                    Person_artday = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_p){
                    Person_p = (float) fgpu_atof(buffer); 
                }
				if(in_Person_q){
                    Person_q = (float) fgpu_atof(buffer); 
                }
				if(in_Person_infections){
                    Person_infections = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_lastinfected){
                    Person_lastinfected = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_lastinfectedid){
                    Person_lastinfectedid = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_lambda){
                    Person_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_Person_timevisiting){
                    Person_timevisiting = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_bargoing){
                    Person_bargoing = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_barday){
                    Person_barday = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_schooltime){
                    Person_schooltime = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_TBAssignment_id){
                    TBAssignment_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_id){
                    Household_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_lambda){
                    Household_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_Household_active){
                    Household_active = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_household_id){
                    HouseholdMembership_household_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_person_id){
                    HouseholdMembership_person_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_household_size){
                    HouseholdMembership_household_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_churchgoing){
                    HouseholdMembership_churchgoing = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_churchfreq){
                    HouseholdMembership_churchfreq = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_id){
                    Church_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_size){
                    Church_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_lambda){
                    Church_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_Church_active){
                    Church_active = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_ChurchMembership_church_id){
                    ChurchMembership_church_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_ChurchMembership_household_id){
                    ChurchMembership_household_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_ChurchMembership_churchdur){
                    ChurchMembership_churchdur = (float) fgpu_atof(buffer); 
                }
				if(in_Transport_id){
                    Transport_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Transport_lambda){
                    Transport_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_Transport_active){
                    Transport_active = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_TransportMembership_person_id){
                    TransportMembership_person_id = (int) fpgu_strtol(buffer); 
                }
				if(in_TransportMembership_transport_id){
                    TransportMembership_transport_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_TransportMembership_duration){
                    TransportMembership_duration = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Clinic_id){
                    Clinic_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Clinic_lambda){
                    Clinic_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_Workplace_id){
                    Workplace_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Workplace_lambda){
                    Workplace_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_WorkplaceMembership_person_id){
                    WorkplaceMembership_person_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_WorkplaceMembership_workplace_id){
                    WorkplaceMembership_workplace_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Bar_id){
                    Bar_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Bar_lambda){
                    Bar_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_School_id){
                    School_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_School_lambda){
                    School_lambda = (float) fgpu_atof(buffer); 
                }
				if(in_SchoolMembership_person_id){
                    SchoolMembership_person_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_SchoolMembership_school_id){
                    SchoolMembership_school_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_TIME_STEP){
              
                    env_TIME_STEP = (float) fgpu_atof(buffer);
                    
                    set_TIME_STEP(&env_TIME_STEP);
                  
              }
            if(in_env_MAX_AGE){
              
                    env_MAX_AGE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_MAX_AGE(&env_MAX_AGE);
                  
              }
            if(in_env_STARTING_POPULATION){
              
                    env_STARTING_POPULATION = (float) fgpu_atof(buffer);
                    
                    set_STARTING_POPULATION(&env_STARTING_POPULATION);
                  
              }
            if(in_env_CHURCH_BETA0){
              
                    env_CHURCH_BETA0 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_BETA0(&env_CHURCH_BETA0);
                  
              }
            if(in_env_CHURCH_BETA1){
              
                    env_CHURCH_BETA1 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_BETA1(&env_CHURCH_BETA1);
                  
              }
            if(in_env_CHURCH_K1){
              
                    env_CHURCH_K1 = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_CHURCH_K1(&env_CHURCH_K1);
                  
              }
            if(in_env_CHURCH_K2){
              
                    env_CHURCH_K2 = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_CHURCH_K2(&env_CHURCH_K2);
                  
              }
            if(in_env_CHURCH_K3){
              
                    env_CHURCH_K3 = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_CHURCH_K3(&env_CHURCH_K3);
                  
              }
            if(in_env_CHURCH_P1){
              
                    env_CHURCH_P1 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_P1(&env_CHURCH_P1);
                  
              }
            if(in_env_CHURCH_P2){
              
                    env_CHURCH_P2 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_P2(&env_CHURCH_P2);
                  
              }
            if(in_env_CHURCH_PROB0){
              
                    env_CHURCH_PROB0 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB0(&env_CHURCH_PROB0);
                  
              }
            if(in_env_CHURCH_PROB1){
              
                    env_CHURCH_PROB1 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB1(&env_CHURCH_PROB1);
                  
              }
            if(in_env_CHURCH_PROB2){
              
                    env_CHURCH_PROB2 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB2(&env_CHURCH_PROB2);
                  
              }
            if(in_env_CHURCH_PROB3){
              
                    env_CHURCH_PROB3 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB3(&env_CHURCH_PROB3);
                  
              }
            if(in_env_CHURCH_PROB4){
              
                    env_CHURCH_PROB4 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB4(&env_CHURCH_PROB4);
                  
              }
            if(in_env_CHURCH_PROB5){
              
                    env_CHURCH_PROB5 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB5(&env_CHURCH_PROB5);
                  
              }
            if(in_env_CHURCH_PROB6){
              
                    env_CHURCH_PROB6 = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_PROB6(&env_CHURCH_PROB6);
                  
              }
            if(in_env_CHURCH_DURATION){
              
                    env_CHURCH_DURATION = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_DURATION(&env_CHURCH_DURATION);
                  
              }
            if(in_env_TRANSPORT_BETA0){
              
                    env_TRANSPORT_BETA0 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_BETA0(&env_TRANSPORT_BETA0);
                  
              }
            if(in_env_TRANSPORT_BETA1){
              
                    env_TRANSPORT_BETA1 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_BETA1(&env_TRANSPORT_BETA1);
                  
              }
            if(in_env_TRANSPORT_FREQ0){
              
                    env_TRANSPORT_FREQ0 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_FREQ0(&env_TRANSPORT_FREQ0);
                  
              }
            if(in_env_TRANSPORT_FREQ2){
              
                    env_TRANSPORT_FREQ2 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_FREQ2(&env_TRANSPORT_FREQ2);
                  
              }
            if(in_env_TRANSPORT_DUR20){
              
                    env_TRANSPORT_DUR20 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_DUR20(&env_TRANSPORT_DUR20);
                  
              }
            if(in_env_TRANSPORT_DUR45){
              
                    env_TRANSPORT_DUR45 = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_DUR45(&env_TRANSPORT_DUR45);
                  
              }
            if(in_env_TRANSPORT_SIZE){
              
                    env_TRANSPORT_SIZE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_TRANSPORT_SIZE(&env_TRANSPORT_SIZE);
                  
              }
            if(in_env_HIV_PREVALENCE){
              
                    env_HIV_PREVALENCE = (float) fgpu_atof(buffer);
                    
                    set_HIV_PREVALENCE(&env_HIV_PREVALENCE);
                  
              }
            if(in_env_ART_COVERAGE){
              
                    env_ART_COVERAGE = (float) fgpu_atof(buffer);
                    
                    set_ART_COVERAGE(&env_ART_COVERAGE);
                  
              }
            if(in_env_RR_HIV){
              
                    env_RR_HIV = (float) fgpu_atof(buffer);
                    
                    set_RR_HIV(&env_RR_HIV);
                  
              }
            if(in_env_RR_ART){
              
                    env_RR_ART = (float) fgpu_atof(buffer);
                    
                    set_RR_ART(&env_RR_ART);
                  
              }
            if(in_env_TB_PREVALENCE){
              
                    env_TB_PREVALENCE = (float) fgpu_atof(buffer);
                    
                    set_TB_PREVALENCE(&env_TB_PREVALENCE);
                  
              }
            if(in_env_DEFAULT_P){
              
                    env_DEFAULT_P = (float) fgpu_atof(buffer);
                    
                    set_DEFAULT_P(&env_DEFAULT_P);
                  
              }
            if(in_env_DEFAULT_Q){
              
                    env_DEFAULT_Q = (float) fgpu_atof(buffer);
                    
                    set_DEFAULT_Q(&env_DEFAULT_Q);
                  
              }
            if(in_env_TRANSPORT_A){
              
                    env_TRANSPORT_A = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_A(&env_TRANSPORT_A);
                  
              }
            if(in_env_CHURCH_A){
              
                    env_CHURCH_A = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_A(&env_CHURCH_A);
                  
              }
            if(in_env_CLINIC_A){
              
                    env_CLINIC_A = (float) fgpu_atof(buffer);
                    
                    set_CLINIC_A(&env_CLINIC_A);
                  
              }
            if(in_env_HOUSEHOLD_A){
              
                    env_HOUSEHOLD_A = (float) fgpu_atof(buffer);
                    
                    set_HOUSEHOLD_A(&env_HOUSEHOLD_A);
                  
              }
            if(in_env_TRANSPORT_V){
              
                    env_TRANSPORT_V = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_V(&env_TRANSPORT_V);
                  
              }
            if(in_env_HOUSEHOLD_V){
              
                    env_HOUSEHOLD_V = (float) fgpu_atof(buffer);
                    
                    set_HOUSEHOLD_V(&env_HOUSEHOLD_V);
                  
              }
            if(in_env_CLINIC_V){
              
                    env_CLINIC_V = (float) fgpu_atof(buffer);
                    
                    set_CLINIC_V(&env_CLINIC_V);
                  
              }
            if(in_env_CHURCH_V_MULTIPLIER){
              
                    env_CHURCH_V_MULTIPLIER = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_V_MULTIPLIER(&env_CHURCH_V_MULTIPLIER);
                  
              }
            if(in_env_WORKPLACE_BETA0){
              
                    env_WORKPLACE_BETA0 = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_BETA0(&env_WORKPLACE_BETA0);
                  
              }
            if(in_env_WORKPLACE_BETAA){
              
                    env_WORKPLACE_BETAA = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_BETAA(&env_WORKPLACE_BETAA);
                  
              }
            if(in_env_WORKPLACE_BETAS){
              
                    env_WORKPLACE_BETAS = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_BETAS(&env_WORKPLACE_BETAS);
                  
              }
            if(in_env_WORKPLACE_BETAAS){
              
                    env_WORKPLACE_BETAAS = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_BETAAS(&env_WORKPLACE_BETAAS);
                  
              }
            if(in_env_WORKPLACE_A){
              
                    env_WORKPLACE_A = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_A(&env_WORKPLACE_A);
                  
              }
            if(in_env_WORKPLACE_DUR){
              
                    env_WORKPLACE_DUR = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_WORKPLACE_DUR(&env_WORKPLACE_DUR);
                  
              }
            if(in_env_WORKPLACE_SIZE){
              
                    env_WORKPLACE_SIZE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_WORKPLACE_SIZE(&env_WORKPLACE_SIZE);
                  
              }
            if(in_env_WORKPLACE_V){
              
                    env_WORKPLACE_V = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_V(&env_WORKPLACE_V);
                  
              }
            if(in_env_HOUSEHOLDS){
              
                    env_HOUSEHOLDS = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_HOUSEHOLDS(&env_HOUSEHOLDS);
                  
              }
            if(in_env_BARS){
              
                    env_BARS = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_BARS(&env_BARS);
                  
              }
            if(in_env_RR_AS_F_46){
              
                    env_RR_AS_F_46 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_F_46(&env_RR_AS_F_46);
                  
              }
            if(in_env_RR_AS_F_26){
              
                    env_RR_AS_F_26 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_F_26(&env_RR_AS_F_26);
                  
              }
            if(in_env_RR_AS_F_18){
              
                    env_RR_AS_F_18 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_F_18(&env_RR_AS_F_18);
                  
              }
            if(in_env_RR_AS_M_46){
              
                    env_RR_AS_M_46 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_M_46(&env_RR_AS_M_46);
                  
              }
            if(in_env_RR_AS_M_26){
              
                    env_RR_AS_M_26 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_M_26(&env_RR_AS_M_26);
                  
              }
            if(in_env_RR_AS_M_18){
              
                    env_RR_AS_M_18 = (float) fgpu_atof(buffer);
                    
                    set_RR_AS_M_18(&env_RR_AS_M_18);
                  
              }
            if(in_env_BAR_BETA0){
              
                    env_BAR_BETA0 = (float) fgpu_atof(buffer);
                    
                    set_BAR_BETA0(&env_BAR_BETA0);
                  
              }
            if(in_env_BAR_BETAA){
              
                    env_BAR_BETAA = (float) fgpu_atof(buffer);
                    
                    set_BAR_BETAA(&env_BAR_BETAA);
                  
              }
            if(in_env_BAR_BETAS){
              
                    env_BAR_BETAS = (float) fgpu_atof(buffer);
                    
                    set_BAR_BETAS(&env_BAR_BETAS);
                  
              }
            if(in_env_BAR_BETAAS){
              
                    env_BAR_BETAAS = (float) fgpu_atof(buffer);
                    
                    set_BAR_BETAAS(&env_BAR_BETAAS);
                  
              }
            if(in_env_BAR_SIZE){
              
                    env_BAR_SIZE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_BAR_SIZE(&env_BAR_SIZE);
                  
              }
            if(in_env_SCHOOL_SIZE){
              
                    env_SCHOOL_SIZE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_SCHOOL_SIZE(&env_SCHOOL_SIZE);
                  
              }
            if(in_env_BAR_A){
              
                    env_BAR_A = (float) fgpu_atof(buffer);
                    
                    set_BAR_A(&env_BAR_A);
                  
              }
            if(in_env_BAR_V){
              
                    env_BAR_V = (float) fgpu_atof(buffer);
                    
                    set_BAR_V(&env_BAR_V);
                  
              }
            if(in_env_SCHOOL_A){
              
                    env_SCHOOL_A = (float) fgpu_atof(buffer);
                    
                    set_SCHOOL_A(&env_SCHOOL_A);
                  
              }
            if(in_env_SCHOOL_V){
              
                    env_SCHOOL_V = (float) fgpu_atof(buffer);
                    
                    set_SCHOOL_V(&env_SCHOOL_V);
                  
              }
            if(in_env_SEED){
              
                    env_SEED = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_SEED(&env_SEED);
                  
              }
            if(in_env_HOUSEHOLD_EXP){
              
                    env_HOUSEHOLD_EXP = (float) fgpu_atof(buffer);
                    
                    set_HOUSEHOLD_EXP(&env_HOUSEHOLD_EXP);
                  
              }
            if(in_env_CHURCH_EXP){
              
                    env_CHURCH_EXP = (float) fgpu_atof(buffer);
                    
                    set_CHURCH_EXP(&env_CHURCH_EXP);
                  
              }
            if(in_env_TRANSPORT_EXP){
              
                    env_TRANSPORT_EXP = (float) fgpu_atof(buffer);
                    
                    set_TRANSPORT_EXP(&env_TRANSPORT_EXP);
                  
              }
            if(in_env_CLINIC_EXP){
              
                    env_CLINIC_EXP = (float) fgpu_atof(buffer);
                    
                    set_CLINIC_EXP(&env_CLINIC_EXP);
                  
              }
            if(in_env_WORKPLACE_EXP){
              
                    env_WORKPLACE_EXP = (float) fgpu_atof(buffer);
                    
                    set_WORKPLACE_EXP(&env_WORKPLACE_EXP);
                  
              }
            if(in_env_BAR_EXP){
              
                    env_BAR_EXP = (float) fgpu_atof(buffer);
                    
                    set_BAR_EXP(&env_BAR_EXP);
                  
              }
            if(in_env_SCHOOL_EXP){
              
                    env_SCHOOL_EXP = (float) fgpu_atof(buffer);
                    
                    set_SCHOOL_EXP(&env_SCHOOL_EXP);
                  
              }
            if(in_env_PROB){
              
                    env_PROB = (float) fgpu_atof(buffer);
                    
                    set_PROB(&env_PROB);
                  
              }
            if(in_env_BAR_M_PROB1){
              
                    env_BAR_M_PROB1 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB1(&env_BAR_M_PROB1);
                  
              }
            if(in_env_BAR_M_PROB2){
              
                    env_BAR_M_PROB2 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB2(&env_BAR_M_PROB2);
                  
              }
            if(in_env_BAR_M_PROB3){
              
                    env_BAR_M_PROB3 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB3(&env_BAR_M_PROB3);
                  
              }
            if(in_env_BAR_M_PROB4){
              
                    env_BAR_M_PROB4 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB4(&env_BAR_M_PROB4);
                  
              }
            if(in_env_BAR_M_PROB5){
              
                    env_BAR_M_PROB5 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB5(&env_BAR_M_PROB5);
                  
              }
            if(in_env_BAR_M_PROB7){
              
                    env_BAR_M_PROB7 = (float) fgpu_atof(buffer);
                    
                    set_BAR_M_PROB7(&env_BAR_M_PROB7);
                  
              }
            if(in_env_BAR_F_PROB1){
              
                    env_BAR_F_PROB1 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB1(&env_BAR_F_PROB1);
                  
              }
            if(in_env_BAR_F_PROB2){
              
                    env_BAR_F_PROB2 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB2(&env_BAR_F_PROB2);
                  
              }
            if(in_env_BAR_F_PROB3){
              
                    env_BAR_F_PROB3 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB3(&env_BAR_F_PROB3);
                  
              }
            if(in_env_BAR_F_PROB4){
              
                    env_BAR_F_PROB4 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB4(&env_BAR_F_PROB4);
                  
              }
            if(in_env_BAR_F_PROB5){
              
                    env_BAR_F_PROB5 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB5(&env_BAR_F_PROB5);
                  
              }
            if(in_env_BAR_F_PROB7){
              
                    env_BAR_F_PROB7 = (float) fgpu_atof(buffer);
                    
                    set_BAR_F_PROB7(&env_BAR_F_PROB7);
                  
              }
            
          }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

