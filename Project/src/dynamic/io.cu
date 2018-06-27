
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Person_list* h_Persons_default, xmachine_memory_Person_list* d_Persons_default, int h_xmachine_memory_Person_default_count,xmachine_memory_Person_list* h_Persons_s2, xmachine_memory_Person_list* d_Persons_s2, int h_xmachine_memory_Person_s2_count)
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
        
		fputs("<dead>", file);
        sprintf(data, "%u", h_Persons_default->dead[i]);
		fputs(data, file);
		fputs("</dead>\n", file);
        
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
        
		fputs("<dead>", file);
        sprintf(data, "%u", h_Persons_s2->dead[i]);
		fputs(data, file);
		fputs("</dead>\n", file);
        
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
}

void readInitialStates(char* inputpath, xmachine_memory_Person_list* h_Persons, int* h_xmachine_memory_Person_count)
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
    int in_Person_dead;
    int in_Person_gender;
    int in_Person_householdsize;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_TIME_STEP;
    
    int in_env_SCALE_FACTOR;
    
    int in_env_MAX_AGE;
    
    int in_env_RANDOM_AGES;
    
	/* set agent count to zero */
	*h_xmachine_memory_Person_count = 0;
	
	/* Variables for initial state data */
	unsigned int Person_id;
	unsigned int Person_age;
	unsigned int Person_dead;
	unsigned int Person_gender;
	unsigned int Person_householdsize;

    /* Variables for environment variables */
    float env_TIME_STEP;
    float env_SCALE_FACTOR;
    unsigned int env_MAX_AGE;
    unsigned int env_RANDOM_AGES;
    


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
	in_Person_dead = 0;
	in_Person_gender = 0;
	in_Person_householdsize = 0;
    in_env_TIME_STEP = 0;
    in_env_SCALE_FACTOR = 0;
    in_env_MAX_AGE = 0;
    in_env_RANDOM_AGES = 0;
	//set all Person values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Person_MAX; k++)
	{	
		h_Persons->id[k] = 0;
		h_Persons->age[k] = 0;
		h_Persons->dead[k] = 0;
		h_Persons->gender[k] = 0;
		h_Persons->householdsize[k] = 0;
	}
	

	/* Default variables for memory */
    Person_id = 0;
    Person_age = 0;
    Person_dead = 0;
    Person_gender = 0;
    Person_householdsize = 0;

    /* Default variables for environment variables */
    env_TIME_STEP = 0;
    env_SCALE_FACTOR = 0;
    env_MAX_AGE = 0;
    env_RANDOM_AGES = 0;
    
    
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
					h_Persons->dead[*h_xmachine_memory_Person_count] = Person_dead;
					h_Persons->gender[*h_xmachine_memory_Person_count] = Person_gender;
					h_Persons->householdsize[*h_xmachine_memory_Person_count] = Person_householdsize;
					(*h_xmachine_memory_Person_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Person_id = 0;
                Person_age = 0;
                Person_dead = 0;
                Person_gender = 0;
                Person_householdsize = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Person_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Person_id = 0;
			if(strcmp(buffer, "age") == 0) in_Person_age = 1;
			if(strcmp(buffer, "/age") == 0) in_Person_age = 0;
			if(strcmp(buffer, "dead") == 0) in_Person_dead = 1;
			if(strcmp(buffer, "/dead") == 0) in_Person_dead = 0;
			if(strcmp(buffer, "gender") == 0) in_Person_gender = 1;
			if(strcmp(buffer, "/gender") == 0) in_Person_gender = 0;
			if(strcmp(buffer, "householdsize") == 0) in_Person_householdsize = 1;
			if(strcmp(buffer, "/householdsize") == 0) in_Person_householdsize = 0;
			
            /* environment variables */
            if(strcmp(buffer, "TIME_STEP") == 0) in_env_TIME_STEP = 1;
            if(strcmp(buffer, "/TIME_STEP") == 0) in_env_TIME_STEP = 0;
			if(strcmp(buffer, "SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 1;
            if(strcmp(buffer, "/SCALE_FACTOR") == 0) in_env_SCALE_FACTOR = 0;
			if(strcmp(buffer, "MAX_AGE") == 0) in_env_MAX_AGE = 1;
            if(strcmp(buffer, "/MAX_AGE") == 0) in_env_MAX_AGE = 0;
			if(strcmp(buffer, "RANDOM_AGES") == 0) in_env_RANDOM_AGES = 1;
            if(strcmp(buffer, "/RANDOM_AGES") == 0) in_env_RANDOM_AGES = 0;
			

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
				if(in_Person_dead){
                    Person_dead = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_gender){
                    Person_gender = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Person_householdsize){
                    Person_householdsize = (unsigned int) fpgu_strtoul(buffer); 
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

