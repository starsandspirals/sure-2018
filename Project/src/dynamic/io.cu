
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count,xmachine_memory_Household_list* h_Households_hhdefault, xmachine_memory_Household_list* d_Households_hhdefault, int h_xmachine_memory_Household_hhdefault_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships_hhmembershipdefault, xmachine_memory_HouseholdMembership_list* d_HouseholdMemberships_hhmembershipdefault, int h_xmachine_memory_HouseholdMembership_hhmembershipdefault_count,xmachine_memory_Church_list* h_Churchs_chudefault, xmachine_memory_Church_list* d_Churchs_chudefault, int h_xmachine_memory_Church_chudefault_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships_chumembershipdefault, xmachine_memory_ChurchMembership_list* d_ChurchMemberships_chumembershipdefault, int h_xmachine_memory_ChurchMembership_chumembershipdefault_count,xmachine_memory_Transport_list* h_Transports_trdefault, xmachine_memory_Transport_list* d_Transports_trdefault, int h_xmachine_memory_Transport_trdefault_count)
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
        
		fputs("<transportuser>", file);
        sprintf(data, "%u", h_Persons_default->transportuser[i]);
		fputs(data, file);
		fputs("</transportuser>\n", file);
        
		fputs("<transportfreq>", file);
        sprintf(data, "%d", h_Persons_default->transportfreq[i]);
		fputs(data, file);
		fputs("</transportfreq>\n", file);
        
		fputs("<transportdur>", file);
        sprintf(data, "%d", h_Persons_default->transportdur[i]);
		fputs(data, file);
		fputs("</transportdur>\n", file);
        
		fputs("<household>", file);
        sprintf(data, "%u", h_Persons_default->household[i]);
		fputs(data, file);
		fputs("</household>\n", file);
        
		fputs("<church>", file);
        sprintf(data, "%d", h_Persons_default->church[i]);
		fputs(data, file);
		fputs("</church>\n", file);
        
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
        
		fputs("<transportuser>", file);
        sprintf(data, "%u", h_Persons_s2->transportuser[i]);
		fputs(data, file);
		fputs("</transportuser>\n", file);
        
		fputs("<transportfreq>", file);
        sprintf(data, "%d", h_Persons_s2->transportfreq[i]);
		fputs(data, file);
		fputs("</transportfreq>\n", file);
        
		fputs("<transportdur>", file);
        sprintf(data, "%d", h_Persons_s2->transportdur[i]);
		fputs(data, file);
		fputs("</transportdur>\n", file);
        
		fputs("<household>", file);
        sprintf(data, "%u", h_Persons_s2->household[i]);
		fputs(data, file);
		fputs("</household>\n", file);
        
		fputs("<church>", file);
        sprintf(data, "%d", h_Persons_s2->church[i]);
		fputs(data, file);
		fputs("</church>\n", file);
        
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
        
		fputs("<step>", file);
        sprintf(data, "%u", h_Households_hhdefault->step[i]);
		fputs(data, file);
		fputs("</step>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_Households_hhdefault->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<people>", file);
        for (int j=0;j<32;j++){
            fprintf(file, "%d", h_Households_hhdefault->people[(j*xmachine_memory_Household_MAX)+i]);
            if(j!=(32-1))
                fprintf(file, ",");
        }
		fputs("</people>\n", file);
        
		fputs("<churchgoing>", file);
        sprintf(data, "%u", h_Households_hhdefault->churchgoing[i]);
		fputs(data, file);
		fputs("</churchgoing>\n", file);
        
		fputs("<churchfreq>", file);
        sprintf(data, "%u", h_Households_hhdefault->churchfreq[i]);
		fputs(data, file);
		fputs("</churchfreq>\n", file);
        
		fputs("<adults>", file);
        sprintf(data, "%u", h_Households_hhdefault->adults[i]);
		fputs(data, file);
		fputs("</adults>\n", file);
        
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
        
		fputs("<step>", file);
        sprintf(data, "%u", h_Churchs_chudefault->step[i]);
		fputs(data, file);
		fputs("</step>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_Churchs_chudefault->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<duration>", file);
        sprintf(data, "%f", h_Churchs_chudefault->duration[i]);
		fputs(data, file);
		fputs("</duration>\n", file);
        
		fputs("<households>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%d", h_Churchs_chudefault->households[(j*xmachine_memory_Church_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</households>\n", file);
        
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
        
		fputs("<step>", file);
        sprintf(data, "%u", h_Transports_trdefault->step[i]);
		fputs(data, file);
		fputs("</step>\n", file);
        
		fputs("<duration>", file);
        sprintf(data, "%u", h_Transports_trdefault->duration[i]);
		fputs(data, file);
		fputs("</duration>\n", file);
        
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
}

void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count,xmachine_memory_Household_list* h_Households, int* h_xmachine_memory_Household_count,xmachine_memory_HouseholdMembership_list* h_HouseholdMemberships, int* h_xmachine_memory_HouseholdMembership_count,xmachine_memory_Church_list* h_Churchs, int* h_xmachine_memory_Church_count,xmachine_memory_ChurchMembership_list* h_ChurchMemberships, int* h_xmachine_memory_ChurchMembership_count,xmachine_memory_Transport_list* h_Transports, int* h_xmachine_memory_Transport_count)
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
    int in_Person_age;
    int in_Person_gender;
    int in_Person_householdsize;
    int in_Person_churchfreq;
    int in_Person_churchdur;
    int in_Person_transportuser;
    int in_Person_transportfreq;
    int in_Person_transportdur;
    int in_Person_household;
    int in_Person_church;
    int in_Person_busy;
    int in_Person_startstep;
    int in_Person_location;
    int in_Person_locationid;
    int in_Household_id;
    int in_Household_step;
    int in_Household_size;
    int in_Household_people;
    int in_Household_churchgoing;
    int in_Household_churchfreq;
    int in_Household_adults;
    int in_HouseholdMembership_household_id;
    int in_HouseholdMembership_person_id;
    int in_HouseholdMembership_churchgoing;
    int in_HouseholdMembership_churchfreq;
    int in_Church_id;
    int in_Church_step;
    int in_Church_size;
    int in_Church_duration;
    int in_Church_households;
    int in_ChurchMembership_church_id;
    int in_ChurchMembership_household_id;
    int in_ChurchMembership_churchdur;
    int in_Transport_id;
    int in_Transport_step;
    int in_Transport_duration;
    
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
    
	/* set agent count to zero */
	*h_xmachine_memory_Person_count = 0;
	*h_xmachine_memory_Household_count = 0;
	*h_xmachine_memory_HouseholdMembership_count = 0;
	*h_xmachine_memory_Church_count = 0;
	*h_xmachine_memory_ChurchMembership_count = 0;
	*h_xmachine_memory_Transport_count = 0;
	
	/* Variables for initial state data */
	unsigned int Person_id;
	unsigned int Person_step;
	unsigned int Person_age;
	unsigned int Person_gender;
	unsigned int Person_householdsize;
	unsigned int Person_churchfreq;
	float Person_churchdur;
	unsigned int Person_transportuser;
	int Person_transportfreq;
	int Person_transportdur;
	unsigned int Person_household;
	int Person_church;
	unsigned int Person_busy;
	unsigned int Person_startstep;
	unsigned int Person_location;
	unsigned int Person_locationid;
	unsigned int Household_id;
	unsigned int Household_step;
	unsigned int Household_size;
    int Household_people[32];
	unsigned int Household_churchgoing;
	unsigned int Household_churchfreq;
	unsigned int Household_adults;
	unsigned int HouseholdMembership_household_id;
	unsigned int HouseholdMembership_person_id;
	unsigned int HouseholdMembership_churchgoing;
	unsigned int HouseholdMembership_churchfreq;
	unsigned int Church_id;
	unsigned int Church_step;
	unsigned int Church_size;
	float Church_duration;
    int Church_households[128];
	unsigned int ChurchMembership_church_id;
	unsigned int ChurchMembership_household_id;
	float ChurchMembership_churchdur;
	unsigned int Transport_id;
	unsigned int Transport_step;
	unsigned int Transport_duration;

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
	in_Person_age = 0;
	in_Person_gender = 0;
	in_Person_householdsize = 0;
	in_Person_churchfreq = 0;
	in_Person_churchdur = 0;
	in_Person_transportuser = 0;
	in_Person_transportfreq = 0;
	in_Person_transportdur = 0;
	in_Person_household = 0;
	in_Person_church = 0;
	in_Person_busy = 0;
	in_Person_startstep = 0;
	in_Person_location = 0;
	in_Person_locationid = 0;
	in_Household_id = 0;
	in_Household_step = 0;
	in_Household_size = 0;
	in_Household_people = 0;
	in_Household_churchgoing = 0;
	in_Household_churchfreq = 0;
	in_Household_adults = 0;
	in_HouseholdMembership_household_id = 0;
	in_HouseholdMembership_person_id = 0;
	in_HouseholdMembership_churchgoing = 0;
	in_HouseholdMembership_churchfreq = 0;
	in_Church_id = 0;
	in_Church_step = 0;
	in_Church_size = 0;
	in_Church_duration = 0;
	in_Church_households = 0;
	in_ChurchMembership_church_id = 0;
	in_ChurchMembership_household_id = 0;
	in_ChurchMembership_churchdur = 0;
	in_Transport_id = 0;
	in_Transport_step = 0;
	in_Transport_duration = 0;
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
	//set all Person values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Person_MAX; k++)
	{	
		h_Persons->id[k] = 0;
		h_Persons->step[k] = 0;
		h_Persons->age[k] = 0;
		h_Persons->gender[k] = 0;
		h_Persons->householdsize[k] = 0;
		h_Persons->churchfreq[k] = 0;
		h_Persons->churchdur[k] = 0;
		h_Persons->transportuser[k] = 0;
		h_Persons->transportfreq[k] = 0;
		h_Persons->transportdur[k] = 0;
		h_Persons->household[k] = 0;
		h_Persons->church[k] = 0;
		h_Persons->busy[k] = 0;
		h_Persons->startstep[k] = 0;
		h_Persons->location[k] = 0;
		h_Persons->locationid[k] = 0;
	}
	
	//set all Household values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Household_MAX; k++)
	{	
		h_Households->id[k] = 0;
		h_Households->step[k] = 0;
		h_Households->size[k] = 0;
        for (i=0;i<32;i++){
            h_Households->people[(i*xmachine_memory_Household_MAX)+k] = 0;
        }
		h_Households->churchgoing[k] = 0;
		h_Households->churchfreq[k] = 0;
		h_Households->adults[k] = 0;
	}
	
	//set all HouseholdMembership values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_HouseholdMembership_MAX; k++)
	{	
		h_HouseholdMemberships->household_id[k] = 0;
		h_HouseholdMemberships->person_id[k] = 0;
		h_HouseholdMemberships->churchgoing[k] = 0;
		h_HouseholdMemberships->churchfreq[k] = 0;
	}
	
	//set all Church values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Church_MAX; k++)
	{	
		h_Churchs->id[k] = 0;
		h_Churchs->step[k] = 0;
		h_Churchs->size[k] = 0;
		h_Churchs->duration[k] = 0;
        for (i=0;i<128;i++){
            h_Churchs->households[(i*xmachine_memory_Church_MAX)+k] = 0;
        }
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
		h_Transports->step[k] = 0;
		h_Transports->duration[k] = 0;
	}
	

	/* Default variables for memory */
    Person_id = 0;
    Person_step = 0;
    Person_age = 0;
    Person_gender = 0;
    Person_householdsize = 0;
    Person_churchfreq = 0;
    Person_churchdur = 0;
    Person_transportuser = 0;
    Person_transportfreq = 0;
    Person_transportdur = 0;
    Person_household = 0;
    Person_church = 0;
    Person_busy = 0;
    Person_startstep = 0;
    Person_location = 0;
    Person_locationid = 0;
    Household_id = 0;
    Household_step = 0;
    Household_size = 0;
    for (i=0;i<32;i++){
        Household_people[i] = -1;
    }
    Household_churchgoing = 0;
    Household_churchfreq = 0;
    Household_adults = 0;
    HouseholdMembership_household_id = 0;
    HouseholdMembership_person_id = 0;
    HouseholdMembership_churchgoing = 0;
    HouseholdMembership_churchfreq = 0;
    Church_id = 0;
    Church_step = 0;
    Church_size = 0;
    Church_duration = 0;
    for (i=0;i<128;i++){
        Church_households[i] = -1;
    }
    ChurchMembership_church_id = 0;
    ChurchMembership_household_id = 0;
    ChurchMembership_churchdur = 0;
    Transport_id = 0;
    Transport_step = 0;
    Transport_duration = 0;

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
					h_Persons->age[*h_xmachine_memory_Person_count] = Person_age;
					h_Persons->gender[*h_xmachine_memory_Person_count] = Person_gender;
					h_Persons->householdsize[*h_xmachine_memory_Person_count] = Person_householdsize;
					h_Persons->churchfreq[*h_xmachine_memory_Person_count] = Person_churchfreq;
					h_Persons->churchdur[*h_xmachine_memory_Person_count] = Person_churchdur;
					h_Persons->transportuser[*h_xmachine_memory_Person_count] = Person_transportuser;
					h_Persons->transportfreq[*h_xmachine_memory_Person_count] = Person_transportfreq;
					h_Persons->transportdur[*h_xmachine_memory_Person_count] = Person_transportdur;
					h_Persons->household[*h_xmachine_memory_Person_count] = Person_household;
					h_Persons->church[*h_xmachine_memory_Person_count] = Person_church;
					h_Persons->busy[*h_xmachine_memory_Person_count] = Person_busy;
					h_Persons->startstep[*h_xmachine_memory_Person_count] = Person_startstep;
					h_Persons->location[*h_xmachine_memory_Person_count] = Person_location;
					h_Persons->locationid[*h_xmachine_memory_Person_count] = Person_locationid;
					(*h_xmachine_memory_Person_count) ++;	
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
					h_Households->step[*h_xmachine_memory_Household_count] = Household_step;
					h_Households->size[*h_xmachine_memory_Household_count] = Household_size;
                    for (int k=0;k<32;k++){
                        h_Households->people[(k*xmachine_memory_Household_MAX)+(*h_xmachine_memory_Household_count)] = Household_people[k];
                    }
					h_Households->churchgoing[*h_xmachine_memory_Household_count] = Household_churchgoing;
					h_Households->churchfreq[*h_xmachine_memory_Household_count] = Household_churchfreq;
					h_Households->adults[*h_xmachine_memory_Household_count] = Household_adults;
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
					h_Churchs->step[*h_xmachine_memory_Church_count] = Church_step;
					h_Churchs->size[*h_xmachine_memory_Church_count] = Church_size;
					h_Churchs->duration[*h_xmachine_memory_Church_count] = Church_duration;
                    for (int k=0;k<128;k++){
                        h_Churchs->households[(k*xmachine_memory_Church_MAX)+(*h_xmachine_memory_Church_count)] = Church_households[k];
                    }
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
					h_Transports->step[*h_xmachine_memory_Transport_count] = Transport_step;
					h_Transports->duration[*h_xmachine_memory_Transport_count] = Transport_duration;
					(*h_xmachine_memory_Transport_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Person_id = 0;
                Person_step = 0;
                Person_age = 0;
                Person_gender = 0;
                Person_householdsize = 0;
                Person_churchfreq = 0;
                Person_churchdur = 0;
                Person_transportuser = 0;
                Person_transportfreq = 0;
                Person_transportdur = 0;
                Person_household = 0;
                Person_church = 0;
                Person_busy = 0;
                Person_startstep = 0;
                Person_location = 0;
                Person_locationid = 0;
                Household_id = 0;
                Household_step = 0;
                Household_size = 0;
                for (i=0;i<32;i++){
                    Household_people[i] = -1;
                }
                Household_churchgoing = 0;
                Household_churchfreq = 0;
                Household_adults = 0;
                HouseholdMembership_household_id = 0;
                HouseholdMembership_person_id = 0;
                HouseholdMembership_churchgoing = 0;
                HouseholdMembership_churchfreq = 0;
                Church_id = 0;
                Church_step = 0;
                Church_size = 0;
                Church_duration = 0;
                for (i=0;i<128;i++){
                    Church_households[i] = -1;
                }
                ChurchMembership_church_id = 0;
                ChurchMembership_household_id = 0;
                ChurchMembership_churchdur = 0;
                Transport_id = 0;
                Transport_step = 0;
                Transport_duration = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Person_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Person_id = 0;
			if(strcmp(buffer, "step") == 0) in_Person_step = 1;
			if(strcmp(buffer, "/step") == 0) in_Person_step = 0;
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
			if(strcmp(buffer, "transportuser") == 0) in_Person_transportuser = 1;
			if(strcmp(buffer, "/transportuser") == 0) in_Person_transportuser = 0;
			if(strcmp(buffer, "transportfreq") == 0) in_Person_transportfreq = 1;
			if(strcmp(buffer, "/transportfreq") == 0) in_Person_transportfreq = 0;
			if(strcmp(buffer, "transportdur") == 0) in_Person_transportdur = 1;
			if(strcmp(buffer, "/transportdur") == 0) in_Person_transportdur = 0;
			if(strcmp(buffer, "household") == 0) in_Person_household = 1;
			if(strcmp(buffer, "/household") == 0) in_Person_household = 0;
			if(strcmp(buffer, "church") == 0) in_Person_church = 1;
			if(strcmp(buffer, "/church") == 0) in_Person_church = 0;
			if(strcmp(buffer, "busy") == 0) in_Person_busy = 1;
			if(strcmp(buffer, "/busy") == 0) in_Person_busy = 0;
			if(strcmp(buffer, "startstep") == 0) in_Person_startstep = 1;
			if(strcmp(buffer, "/startstep") == 0) in_Person_startstep = 0;
			if(strcmp(buffer, "location") == 0) in_Person_location = 1;
			if(strcmp(buffer, "/location") == 0) in_Person_location = 0;
			if(strcmp(buffer, "locationid") == 0) in_Person_locationid = 1;
			if(strcmp(buffer, "/locationid") == 0) in_Person_locationid = 0;
			if(strcmp(buffer, "id") == 0) in_Household_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Household_id = 0;
			if(strcmp(buffer, "step") == 0) in_Household_step = 1;
			if(strcmp(buffer, "/step") == 0) in_Household_step = 0;
			if(strcmp(buffer, "size") == 0) in_Household_size = 1;
			if(strcmp(buffer, "/size") == 0) in_Household_size = 0;
			if(strcmp(buffer, "people") == 0) in_Household_people = 1;
			if(strcmp(buffer, "/people") == 0) in_Household_people = 0;
			if(strcmp(buffer, "churchgoing") == 0) in_Household_churchgoing = 1;
			if(strcmp(buffer, "/churchgoing") == 0) in_Household_churchgoing = 0;
			if(strcmp(buffer, "churchfreq") == 0) in_Household_churchfreq = 1;
			if(strcmp(buffer, "/churchfreq") == 0) in_Household_churchfreq = 0;
			if(strcmp(buffer, "adults") == 0) in_Household_adults = 1;
			if(strcmp(buffer, "/adults") == 0) in_Household_adults = 0;
			if(strcmp(buffer, "household_id") == 0) in_HouseholdMembership_household_id = 1;
			if(strcmp(buffer, "/household_id") == 0) in_HouseholdMembership_household_id = 0;
			if(strcmp(buffer, "person_id") == 0) in_HouseholdMembership_person_id = 1;
			if(strcmp(buffer, "/person_id") == 0) in_HouseholdMembership_person_id = 0;
			if(strcmp(buffer, "churchgoing") == 0) in_HouseholdMembership_churchgoing = 1;
			if(strcmp(buffer, "/churchgoing") == 0) in_HouseholdMembership_churchgoing = 0;
			if(strcmp(buffer, "churchfreq") == 0) in_HouseholdMembership_churchfreq = 1;
			if(strcmp(buffer, "/churchfreq") == 0) in_HouseholdMembership_churchfreq = 0;
			if(strcmp(buffer, "id") == 0) in_Church_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Church_id = 0;
			if(strcmp(buffer, "step") == 0) in_Church_step = 1;
			if(strcmp(buffer, "/step") == 0) in_Church_step = 0;
			if(strcmp(buffer, "size") == 0) in_Church_size = 1;
			if(strcmp(buffer, "/size") == 0) in_Church_size = 0;
			if(strcmp(buffer, "duration") == 0) in_Church_duration = 1;
			if(strcmp(buffer, "/duration") == 0) in_Church_duration = 0;
			if(strcmp(buffer, "households") == 0) in_Church_households = 1;
			if(strcmp(buffer, "/households") == 0) in_Church_households = 0;
			if(strcmp(buffer, "church_id") == 0) in_ChurchMembership_church_id = 1;
			if(strcmp(buffer, "/church_id") == 0) in_ChurchMembership_church_id = 0;
			if(strcmp(buffer, "household_id") == 0) in_ChurchMembership_household_id = 1;
			if(strcmp(buffer, "/household_id") == 0) in_ChurchMembership_household_id = 0;
			if(strcmp(buffer, "churchdur") == 0) in_ChurchMembership_churchdur = 1;
			if(strcmp(buffer, "/churchdur") == 0) in_ChurchMembership_churchdur = 0;
			if(strcmp(buffer, "id") == 0) in_Transport_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Transport_id = 0;
			if(strcmp(buffer, "step") == 0) in_Transport_step = 1;
			if(strcmp(buffer, "/step") == 0) in_Transport_step = 0;
			if(strcmp(buffer, "duration") == 0) in_Transport_duration = 1;
			if(strcmp(buffer, "/duration") == 0) in_Transport_duration = 0;
			
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
				if(in_Person_transportuser){
                    Person_transportuser = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_transportfreq){
                    Person_transportfreq = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_transportdur){
                    Person_transportdur = (int) fpgu_strtol(buffer); 
                }
				if(in_Person_household){
                    Person_household = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_church){
                    Person_church = (int) fpgu_strtol(buffer); 
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
				if(in_Household_id){
                    Household_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_step){
                    Household_step = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_size){
                    Household_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_people){
                    readArrayInput<int>(&fpgu_strtol, buffer, Household_people, 32);    
                }
				if(in_Household_churchgoing){
                    Household_churchgoing = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_churchfreq){
                    Household_churchfreq = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_adults){
                    Household_adults = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_household_id){
                    HouseholdMembership_household_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_HouseholdMembership_person_id){
                    HouseholdMembership_person_id = (unsigned int) fpgu_strtoul(buffer); 
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
				if(in_Church_step){
                    Church_step = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_size){
                    Church_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_duration){
                    Church_duration = (float) fgpu_atof(buffer); 
                }
				if(in_Church_households){
                    readArrayInput<int>(&fpgu_strtol, buffer, Church_households, 128);    
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
				if(in_Transport_step){
                    Transport_step = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Transport_duration){
                    Transport_duration = (unsigned int) fpgu_strtoul(buffer); 
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

