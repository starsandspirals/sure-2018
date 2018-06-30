
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count,xmachine_memory_Household_list* h_Households_hhdefault, xmachine_memory_Household_list* d_Households_hhdefault, int h_xmachine_memory_Household_hhdefault_count,xmachine_memory_Church_list* h_Churchs_chudefault, xmachine_memory_Church_list* d_Churchs_chudefault, int h_xmachine_memory_Church_chudefault_count)
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
	cudaStatus = cudaMemcpy( h_Churchs_chudefault, d_Churchs_chudefault, sizeof(xmachine_memory_Church_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Church Agent chudefault State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    fputs("\t<SCALE_FACTOR>", file);
    sprintf(data, "%f", (*get_SCALE_FACTOR()));
    fputs(data, file);
    fputs("</SCALE_FACTOR>\n", file);
    fputs("\t<MAX_AGE>", file);
    sprintf(data, "%u", (*get_MAX_AGE()));
    fputs(data, file);
    fputs("</MAX_AGE>\n", file);
    fputs("\t<RANDOM_AGES>", file);
    sprintf(data, "%u", (*get_RANDOM_AGES()));
    fputs(data, file);
    fputs("</RANDOM_AGES>\n", file);
    fputs("\t<STARTING_POPULATION>", file);
    sprintf(data, "%f", (*get_STARTING_POPULATION()));
    fputs(data, file);
    fputs("</STARTING_POPULATION>\n", file);
	fputs("</environment>\n" , file);

	//Write each Person agent to xml
	for (int i=0; i<h_xmachine_memory_Person_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Person</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Persons_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
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
        
		fputs("<duration>", file);
        sprintf(data, "%u", h_Churchs_chudefault->duration[i]);
		fputs(data, file);
		fputs("</duration>\n", file);
        
		fputs("<households>", file);
        for (int j=0;j<64;j++){
            fprintf(file, "%d", h_Churchs_chudefault->households[(j*xmachine_memory_Church_MAX)+i]);
            if(j!=(64-1))
                fprintf(file, ",");
        }
		fputs("</households>\n", file);
        
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
    float t_SCALE_FACTOR = (float)0.01;
    set_SCALE_FACTOR(&t_SCALE_FACTOR);
    unsigned int t_MAX_AGE = (unsigned int)100;
    set_MAX_AGE(&t_MAX_AGE);
    unsigned int t_RANDOM_AGES = (unsigned int)1;
    set_RANDOM_AGES(&t_RANDOM_AGES);
    float t_STARTING_POPULATION = (float)30000.0;
    set_STARTING_POPULATION(&t_STARTING_POPULATION);
}

void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count,xmachine_memory_Household_list* h_Households, int* h_xmachine_memory_Household_count,xmachine_memory_Church_list* h_Churchs, int* h_xmachine_memory_Church_count)
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
    int in_Person_age;
    int in_Person_gender;
    int in_Person_householdsize;
    int in_Household_id;
    int in_Household_size;
    int in_Household_people;
    int in_Household_churchgoing;
    int in_Household_churchfreq;
    int in_Household_adults;
    int in_Church_id;
    int in_Church_size;
    int in_Church_duration;
    int in_Church_households;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_TIME_STEP;
    
    int in_env_SCALE_FACTOR;
    
    int in_env_MAX_AGE;
    
    int in_env_RANDOM_AGES;
    
    int in_env_STARTING_POPULATION;
    
	/* set agent count to zero */
	*h_xmachine_memory_Person_count = 0;
	*h_xmachine_memory_Household_count = 0;
	*h_xmachine_memory_Church_count = 0;
	
	/* Variables for initial state data */
	unsigned int Person_id;
	unsigned int Person_age;
	unsigned int Person_gender;
	unsigned int Person_householdsize;
	unsigned int Household_id;
	unsigned int Household_size;
    int Household_people[32];
	unsigned int Household_churchgoing;
	unsigned int Household_churchfreq;
	unsigned int Household_adults;
	unsigned int Church_id;
	unsigned int Church_size;
	unsigned int Church_duration;
    int Church_households[64];

    /* Variables for environment variables */
    float env_TIME_STEP;
    float env_SCALE_FACTOR;
    unsigned int env_MAX_AGE;
    unsigned int env_RANDOM_AGES;
    float env_STARTING_POPULATION;
    


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
	in_Person_age = 0;
	in_Person_gender = 0;
	in_Person_householdsize = 0;
	in_Household_id = 0;
	in_Household_size = 0;
	in_Household_people = 0;
	in_Household_churchgoing = 0;
	in_Household_churchfreq = 0;
	in_Household_adults = 0;
	in_Church_id = 0;
	in_Church_size = 0;
	in_Church_duration = 0;
	in_Church_households = 0;
    in_env_TIME_STEP = 0;
    in_env_SCALE_FACTOR = 0;
    in_env_MAX_AGE = 0;
    in_env_RANDOM_AGES = 0;
    in_env_STARTING_POPULATION = 0;
	//set all Person values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Person_MAX; k++)
	{	
		h_Persons->id[k] = 0;
		h_Persons->age[k] = 0;
		h_Persons->gender[k] = 0;
		h_Persons->householdsize[k] = 0;
	}
	
	//set all Household values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Household_MAX; k++)
	{	
		h_Households->id[k] = 0;
		h_Households->size[k] = 0;
        for (i=0;i<32;i++){
            h_Households->people[(i*xmachine_memory_Household_MAX)+k] = 0;
        }
		h_Households->churchgoing[k] = 0;
		h_Households->churchfreq[k] = 0;
		h_Households->adults[k] = 0;
	}
	
	//set all Church values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Church_MAX; k++)
	{	
		h_Churchs->id[k] = 0;
		h_Churchs->size[k] = 0;
		h_Churchs->duration[k] = 0;
        for (i=0;i<64;i++){
            h_Churchs->households[(i*xmachine_memory_Church_MAX)+k] = 0;
        }
	}
	

	/* Default variables for memory */
    Person_id = 0;
    Person_age = 0;
    Person_gender = 0;
    Person_householdsize = 0;
    Household_id = 0;
    Household_size = 0;
    for (i=0;i<32;i++){
        Household_people[i] = -1;
    }
    Household_churchgoing = 0;
    Household_churchfreq = 0;
    Household_adults = 0;
    Church_id = 0;
    Church_size = 0;
    Church_duration = 0;
    for (i=0;i<64;i++){
        Church_households[i] = -1;
    }

    /* Default variables for environment variables */
    env_TIME_STEP = 0;
    env_SCALE_FACTOR = 0;
    env_MAX_AGE = 0;
    env_RANDOM_AGES = 0;
    env_STARTING_POPULATION = 0;
    
    
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
					h_Persons->age[*h_xmachine_memory_Person_count] = Person_age;
					h_Persons->gender[*h_xmachine_memory_Person_count] = Person_gender;
					h_Persons->householdsize[*h_xmachine_memory_Person_count] = Person_householdsize;
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
					h_Households->size[*h_xmachine_memory_Household_count] = Household_size;
                    for (int k=0;k<32;k++){
                        h_Households->people[(k*xmachine_memory_Household_MAX)+(*h_xmachine_memory_Household_count)] = Household_people[k];
                    }
					h_Households->churchgoing[*h_xmachine_memory_Household_count] = Household_churchgoing;
					h_Households->churchfreq[*h_xmachine_memory_Household_count] = Household_churchfreq;
					h_Households->adults[*h_xmachine_memory_Household_count] = Household_adults;
					(*h_xmachine_memory_Household_count) ++;	
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
					h_Churchs->duration[*h_xmachine_memory_Church_count] = Church_duration;
                    for (int k=0;k<64;k++){
                        h_Churchs->households[(k*xmachine_memory_Church_MAX)+(*h_xmachine_memory_Church_count)] = Church_households[k];
                    }
					(*h_xmachine_memory_Church_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Person_id = 0;
                Person_age = 0;
                Person_gender = 0;
                Person_householdsize = 0;
                Household_id = 0;
                Household_size = 0;
                for (i=0;i<32;i++){
                    Household_people[i] = -1;
                }
                Household_churchgoing = 0;
                Household_churchfreq = 0;
                Household_adults = 0;
                Church_id = 0;
                Church_size = 0;
                Church_duration = 0;
                for (i=0;i<64;i++){
                    Church_households[i] = -1;
                }
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Person_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Person_id = 0;
			if(strcmp(buffer, "age") == 0) in_Person_age = 1;
			if(strcmp(buffer, "/age") == 0) in_Person_age = 0;
			if(strcmp(buffer, "gender") == 0) in_Person_gender = 1;
			if(strcmp(buffer, "/gender") == 0) in_Person_gender = 0;
			if(strcmp(buffer, "householdsize") == 0) in_Person_householdsize = 1;
			if(strcmp(buffer, "/householdsize") == 0) in_Person_householdsize = 0;
			if(strcmp(buffer, "id") == 0) in_Household_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Household_id = 0;
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
			if(strcmp(buffer, "id") == 0) in_Church_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Church_id = 0;
			if(strcmp(buffer, "size") == 0) in_Church_size = 1;
			if(strcmp(buffer, "/size") == 0) in_Church_size = 0;
			if(strcmp(buffer, "duration") == 0) in_Church_duration = 1;
			if(strcmp(buffer, "/duration") == 0) in_Church_duration = 0;
			if(strcmp(buffer, "households") == 0) in_Church_households = 1;
			if(strcmp(buffer, "/households") == 0) in_Church_households = 0;
			
            /* environment variables */
            if(strcmp(buffer, "TIME_STEP") == 0) in_env_TIME_STEP = 1;
            if(strcmp(buffer, "/TIME_STEP") == 0) in_env_TIME_STEP = 0;
			if(strcmp(buffer, "SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 1;
            if(strcmp(buffer, "/SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 0;
			if(strcmp(buffer, "MAX_AGE") == 0) in_env_MAX_AGE = 1;
            if(strcmp(buffer, "/MAX_AGE") == 0) in_env_MAX_AGE = 0;
			if(strcmp(buffer, "RANDOM_AGES") == 0) in_env_RANDOM_AGES = 1;
            if(strcmp(buffer, "/RANDOM_AGES") == 0) in_env_RANDOM_AGES = 0;
			if(strcmp(buffer, "STARTING_POPULATION") == 0) in_env_STARTING_POPULATION = 1;
            if(strcmp(buffer, "/STARTING_POPULATION") == 0) in_env_STARTING_POPULATION = 0;
			

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
				if(in_Person_age){
                    Person_age = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_gender){
                    Person_gender = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_householdsize){
                    Person_householdsize = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Household_id){
                    Household_id = (unsigned int) fpgu_strtoul(buffer); 
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
				if(in_Church_id){
                    Church_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_size){
                    Church_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_duration){
                    Church_duration = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Church_households){
                    readArrayInput<int>(&fpgu_strtol, buffer, Church_households, 64);    
                }
				
            }
            else if (in_env){
            if(in_env_TIME_STEP){
              
                    env_TIME_STEP = (float) fgpu_atof(buffer);
                    
                    set_TIME_STEP(&env_TIME_STEP);
                  
              }
            if(in_env_SCALE_FACTOR){
              
                    env_SCALE_FACTOR = (float) fgpu_atof(buffer);
                    
                    set_SCALE_FACTOR(&env_SCALE_FACTOR);
                  
              }
            if(in_env_MAX_AGE){
              
                    env_MAX_AGE = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_MAX_AGE(&env_MAX_AGE);
                  
              }
            if(in_env_RANDOM_AGES){
              
                    env_RANDOM_AGES = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_RANDOM_AGES(&env_RANDOM_AGES);
                  
              }
            if(in_env_STARTING_POPULATION){
              
                    env_STARTING_POPULATION = (float) fgpu_atof(buffer);
                    
                    set_STARTING_POPULATION(&env_STARTING_POPULATION);
                  
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

