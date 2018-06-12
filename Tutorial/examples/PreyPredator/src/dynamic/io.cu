
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
	

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_prey_list* h_preys_default1, xmachine_memory_prey_list* d_preys_default1, int h_xmachine_memory_prey_default1_count,xmachine_memory_predator_list* h_predators_default2, xmachine_memory_predator_list* d_predators_default2, int h_xmachine_memory_predator_default2_count,xmachine_memory_grass_list* h_grasss_default3, xmachine_memory_grass_list* d_grasss_default3, int h_xmachine_memory_grass_default3_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_preys_default1, d_preys_default1, sizeof(xmachine_memory_prey_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying prey Agent default1 State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_predators_default2, d_predators_default2, sizeof(xmachine_memory_predator_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying predator Agent default2 State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_grasss_default3, d_grasss_default3, sizeof(xmachine_memory_grass_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying grass Agent default3 State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each prey agent to xml
	for (int i=0; i<h_xmachine_memory_prey_default1_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>prey</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_preys_default1->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_preys_default1->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_preys_default1->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%f", h_preys_default1->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_preys_default1->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_preys_default1->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<steer_x>", file);
        sprintf(data, "%f", h_preys_default1->steer_x[i]);
		fputs(data, file);
		fputs("</steer_x>\n", file);
        
		fputs("<steer_y>", file);
        sprintf(data, "%f", h_preys_default1->steer_y[i]);
		fputs(data, file);
		fputs("</steer_y>\n", file);
        
		fputs("<life>", file);
        sprintf(data, "%i", h_preys_default1->life[i]);
		fputs(data, file);
		fputs("</life>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each predator agent to xml
	for (int i=0; i<h_xmachine_memory_predator_default2_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>predator</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_predators_default2->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_predators_default2->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_predators_default2->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%f", h_predators_default2->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_predators_default2->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_predators_default2->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("<steer_x>", file);
        sprintf(data, "%f", h_predators_default2->steer_x[i]);
		fputs(data, file);
		fputs("</steer_x>\n", file);
        
		fputs("<steer_y>", file);
        sprintf(data, "%f", h_predators_default2->steer_y[i]);
		fputs(data, file);
		fputs("</steer_y>\n", file);
        
		fputs("<life>", file);
        sprintf(data, "%i", h_predators_default2->life[i]);
		fputs(data, file);
		fputs("</life>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each grass agent to xml
	for (int i=0; i<h_xmachine_memory_grass_default3_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>grass</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_grasss_default3->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_grasss_default3->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_grasss_default3->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%f", h_grasss_default3->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<dead_cycles>", file);
        sprintf(data, "%i", h_grasss_default3->dead_cycles[i]);
		fputs(data, file);
		fputs("</dead_cycles>\n", file);
        
		fputs("<available>", file);
        sprintf(data, "%i", h_grasss_default3->available[i]);
		fputs(data, file);
		fputs("</available>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void initEnvVars()
{
    float t_REPRODUCE_PREY_PROB = (float)0.05;
    set_REPRODUCE_PREY_PROB(&t_REPRODUCE_PREY_PROB);
    float t_REPRODUCE_PREDATOR_PROB = (float)0.03;
    set_REPRODUCE_PREDATOR_PROB(&t_REPRODUCE_PREDATOR_PROB);
    int t_GAIN_FROM_FOOD_PREDATOR = (int)75;
    set_GAIN_FROM_FOOD_PREDATOR(&t_GAIN_FROM_FOOD_PREDATOR);
    int t_GAIN_FROM_FOOD_PREY = (int)50;
    set_GAIN_FROM_FOOD_PREY(&t_GAIN_FROM_FOOD_PREY);
    int t_GRASS_REGROW_CYCLES = (int)100;
    set_GRASS_REGROW_CYCLES(&t_GRASS_REGROW_CYCLES);
}
void readInitialStates(char* inputpath, xmachine_memory_prey_list* h_preys, int* h_xmachine_memory_prey_count,xmachine_memory_predator_list* h_predators, int* h_xmachine_memory_predator_count,xmachine_memory_grass_list* h_grasss, int* h_xmachine_memory_grass_count)
{

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
    int in_prey_id;
    int in_prey_x;
    int in_prey_y;
    int in_prey_type;
    int in_prey_fx;
    int in_prey_fy;
    int in_prey_steer_x;
    int in_prey_steer_y;
    int in_prey_life;
    int in_predator_id;
    int in_predator_x;
    int in_predator_y;
    int in_predator_type;
    int in_predator_fx;
    int in_predator_fy;
    int in_predator_steer_x;
    int in_predator_steer_y;
    int in_predator_life;
    int in_grass_id;
    int in_grass_x;
    int in_grass_y;
    int in_grass_type;
    int in_grass_dead_cycles;
    int in_grass_available;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_REPRODUCE_PREY_PROB;
    
    int in_env_REPRODUCE_PREDATOR_PROB;
    
    int in_env_GAIN_FROM_FOOD_PREDATOR;
    
    int in_env_GAIN_FROM_FOOD_PREY;
    
    int in_env_GRASS_REGROW_CYCLES;
    
    

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_prey_count = 0;	
	*h_xmachine_memory_predator_count = 0;	
	*h_xmachine_memory_grass_count = 0;
	
	/* Variables for initial state data */
	int prey_id;
	float prey_x;
	float prey_y;
	float prey_type;
	float prey_fx;
	float prey_fy;
	float prey_steer_x;
	float prey_steer_y;
	int prey_life;
	int predator_id;
	float predator_x;
	float predator_y;
	float predator_type;
	float predator_fx;
	float predator_fy;
	float predator_steer_x;
	float predator_steer_y;
	int predator_life;
	int grass_id;
	float grass_x;
	float grass_y;
	float grass_type;
	int grass_dead_cycles;
	int grass_available;
    
    /* Variables for environment variables */
    float env_REPRODUCE_PREY_PROB;   
    float env_REPRODUCE_PREDATOR_PROB;   
    int env_GAIN_FROM_FOOD_PREDATOR;   
    int env_GAIN_FROM_FOOD_PREY;   
    int env_GRASS_REGROW_CYCLES;   
    
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("Error opening initial states\n");
		exit(EXIT_FAILURE);
	}
	
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
	in_prey_id = 0;
	in_prey_x = 0;
	in_prey_y = 0;
	in_prey_type = 0;
	in_prey_fx = 0;
	in_prey_fy = 0;
	in_prey_steer_x = 0;
	in_prey_steer_y = 0;
	in_prey_life = 0;
	in_predator_id = 0;
	in_predator_x = 0;
	in_predator_y = 0;
	in_predator_type = 0;
	in_predator_fx = 0;
	in_predator_fy = 0;
	in_predator_steer_x = 0;
	in_predator_steer_y = 0;
	in_predator_life = 0;
	in_grass_id = 0;
	in_grass_x = 0;
	in_grass_y = 0;
	in_grass_type = 0;
	in_grass_dead_cycles = 0;
	in_grass_available = 0;
    in_env_REPRODUCE_PREY_PROB = 0;
    in_env_REPRODUCE_PREDATOR_PROB = 0;
    in_env_GAIN_FROM_FOOD_PREDATOR = 0;
    in_env_GAIN_FROM_FOOD_PREY = 0;
    in_env_GRASS_REGROW_CYCLES = 0;
	//set all prey values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_prey_MAX; k++)
	{	
		h_preys->id[k] = 0;
		h_preys->x[k] = 0;
		h_preys->y[k] = 0;
		h_preys->type[k] = 0;
		h_preys->fx[k] = 0;
		h_preys->fy[k] = 0;
		h_preys->steer_x[k] = 0;
		h_preys->steer_y[k] = 0;
		h_preys->life[k] = 0;
	}
	
	//set all predator values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_predator_MAX; k++)
	{	
		h_predators->id[k] = 0;
		h_predators->x[k] = 0;
		h_predators->y[k] = 0;
		h_predators->type[k] = 0;
		h_predators->fx[k] = 0;
		h_predators->fy[k] = 0;
		h_predators->steer_x[k] = 0;
		h_predators->steer_y[k] = 0;
		h_predators->life[k] = 0;
	}
	
	//set all grass values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_grass_MAX; k++)
	{	
		h_grasss->id[k] = 0;
		h_grasss->x[k] = 0;
		h_grasss->y[k] = 0;
		h_grasss->type[k] = 0;
		h_grasss->dead_cycles[k] = 0;
		h_grasss->available[k] = 0;
	}
	

	/* Default variables for memory */
    prey_id = 0;
    prey_x = 0;
    prey_y = 0;
    prey_type = 0;
    prey_fx = 0;
    prey_fy = 0;
    prey_steer_x = 0;
    prey_steer_y = 0;
    prey_life = 0;
    predator_id = 0;
    predator_x = 0;
    predator_y = 0;
    predator_type = 0;
    predator_fx = 0;
    predator_fy = 0;
    predator_steer_x = 0;
    predator_steer_y = 0;
    predator_life = 0;
    grass_id = 0;
    grass_x = 0;
    grass_y = 0;
    grass_type = 0;
    grass_dead_cycles = 0;
    grass_available = 0;

    /* Default variables for environment variables */
    env_REPRODUCE_PREY_PROB = 0;
    env_REPRODUCE_PREDATOR_PROB = 0;
    env_GAIN_FROM_FOOD_PREDATOR = 0;
    env_GAIN_FROM_FOOD_PREY = 0;
    env_GRASS_REGROW_CYCLES = 0;
    
    
	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
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
				if(strcmp(agentname, "prey") == 0)
				{		
					if (*h_xmachine_memory_prey_count > xmachine_memory_prey_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent prey exceeded whilst reading data\n", xmachine_memory_prey_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_preys->id[*h_xmachine_memory_prey_count] = prey_id;
					h_preys->x[*h_xmachine_memory_prey_count] = prey_x;//Check maximum x value
                    if(agent_maximum.x < prey_x)
                        agent_maximum.x = (float)prey_x;
                    //Check minimum x value
                    if(agent_minimum.x > prey_x)
                        agent_minimum.x = (float)prey_x;
                    
					h_preys->y[*h_xmachine_memory_prey_count] = prey_y;//Check maximum y value
                    if(agent_maximum.y < prey_y)
                        agent_maximum.y = (float)prey_y;
                    //Check minimum y value
                    if(agent_minimum.y > prey_y)
                        agent_minimum.y = (float)prey_y;
                    
					h_preys->type[*h_xmachine_memory_prey_count] = prey_type;
					h_preys->fx[*h_xmachine_memory_prey_count] = prey_fx;
					h_preys->fy[*h_xmachine_memory_prey_count] = prey_fy;
					h_preys->steer_x[*h_xmachine_memory_prey_count] = prey_steer_x;
					h_preys->steer_y[*h_xmachine_memory_prey_count] = prey_steer_y;
					h_preys->life[*h_xmachine_memory_prey_count] = prey_life;
					(*h_xmachine_memory_prey_count) ++;	
				}
				else if(strcmp(agentname, "predator") == 0)
				{		
					if (*h_xmachine_memory_predator_count > xmachine_memory_predator_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent predator exceeded whilst reading data\n", xmachine_memory_predator_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_predators->id[*h_xmachine_memory_predator_count] = predator_id;
					h_predators->x[*h_xmachine_memory_predator_count] = predator_x;//Check maximum x value
                    if(agent_maximum.x < predator_x)
                        agent_maximum.x = (float)predator_x;
                    //Check minimum x value
                    if(agent_minimum.x > predator_x)
                        agent_minimum.x = (float)predator_x;
                    
					h_predators->y[*h_xmachine_memory_predator_count] = predator_y;//Check maximum y value
                    if(agent_maximum.y < predator_y)
                        agent_maximum.y = (float)predator_y;
                    //Check minimum y value
                    if(agent_minimum.y > predator_y)
                        agent_minimum.y = (float)predator_y;
                    
					h_predators->type[*h_xmachine_memory_predator_count] = predator_type;
					h_predators->fx[*h_xmachine_memory_predator_count] = predator_fx;
					h_predators->fy[*h_xmachine_memory_predator_count] = predator_fy;
					h_predators->steer_x[*h_xmachine_memory_predator_count] = predator_steer_x;
					h_predators->steer_y[*h_xmachine_memory_predator_count] = predator_steer_y;
					h_predators->life[*h_xmachine_memory_predator_count] = predator_life;
					(*h_xmachine_memory_predator_count) ++;	
				}
				else if(strcmp(agentname, "grass") == 0)
				{		
					if (*h_xmachine_memory_grass_count > xmachine_memory_grass_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent grass exceeded whilst reading data\n", xmachine_memory_grass_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_grasss->id[*h_xmachine_memory_grass_count] = grass_id;
					h_grasss->x[*h_xmachine_memory_grass_count] = grass_x;//Check maximum x value
                    if(agent_maximum.x < grass_x)
                        agent_maximum.x = (float)grass_x;
                    //Check minimum x value
                    if(agent_minimum.x > grass_x)
                        agent_minimum.x = (float)grass_x;
                    
					h_grasss->y[*h_xmachine_memory_grass_count] = grass_y;//Check maximum y value
                    if(agent_maximum.y < grass_y)
                        agent_maximum.y = (float)grass_y;
                    //Check minimum y value
                    if(agent_minimum.y > grass_y)
                        agent_minimum.y = (float)grass_y;
                    
					h_grasss->type[*h_xmachine_memory_grass_count] = grass_type;
					h_grasss->dead_cycles[*h_xmachine_memory_grass_count] = grass_dead_cycles;
					h_grasss->available[*h_xmachine_memory_grass_count] = grass_available;
					(*h_xmachine_memory_grass_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                prey_id = 0;
                prey_x = 0;
                prey_y = 0;
                prey_type = 0;
                prey_fx = 0;
                prey_fy = 0;
                prey_steer_x = 0;
                prey_steer_y = 0;
                prey_life = 0;
                predator_id = 0;
                predator_x = 0;
                predator_y = 0;
                predator_type = 0;
                predator_fx = 0;
                predator_fy = 0;
                predator_steer_x = 0;
                predator_steer_y = 0;
                predator_life = 0;
                grass_id = 0;
                grass_x = 0;
                grass_y = 0;
                grass_type = 0;
                grass_dead_cycles = 0;
                grass_available = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_prey_id = 1;
			if(strcmp(buffer, "/id") == 0) in_prey_id = 0;
			if(strcmp(buffer, "x") == 0) in_prey_x = 1;
			if(strcmp(buffer, "/x") == 0) in_prey_x = 0;
			if(strcmp(buffer, "y") == 0) in_prey_y = 1;
			if(strcmp(buffer, "/y") == 0) in_prey_y = 0;
			if(strcmp(buffer, "type") == 0) in_prey_type = 1;
			if(strcmp(buffer, "/type") == 0) in_prey_type = 0;
			if(strcmp(buffer, "fx") == 0) in_prey_fx = 1;
			if(strcmp(buffer, "/fx") == 0) in_prey_fx = 0;
			if(strcmp(buffer, "fy") == 0) in_prey_fy = 1;
			if(strcmp(buffer, "/fy") == 0) in_prey_fy = 0;
			if(strcmp(buffer, "steer_x") == 0) in_prey_steer_x = 1;
			if(strcmp(buffer, "/steer_x") == 0) in_prey_steer_x = 0;
			if(strcmp(buffer, "steer_y") == 0) in_prey_steer_y = 1;
			if(strcmp(buffer, "/steer_y") == 0) in_prey_steer_y = 0;
			if(strcmp(buffer, "life") == 0) in_prey_life = 1;
			if(strcmp(buffer, "/life") == 0) in_prey_life = 0;
			if(strcmp(buffer, "id") == 0) in_predator_id = 1;
			if(strcmp(buffer, "/id") == 0) in_predator_id = 0;
			if(strcmp(buffer, "x") == 0) in_predator_x = 1;
			if(strcmp(buffer, "/x") == 0) in_predator_x = 0;
			if(strcmp(buffer, "y") == 0) in_predator_y = 1;
			if(strcmp(buffer, "/y") == 0) in_predator_y = 0;
			if(strcmp(buffer, "type") == 0) in_predator_type = 1;
			if(strcmp(buffer, "/type") == 0) in_predator_type = 0;
			if(strcmp(buffer, "fx") == 0) in_predator_fx = 1;
			if(strcmp(buffer, "/fx") == 0) in_predator_fx = 0;
			if(strcmp(buffer, "fy") == 0) in_predator_fy = 1;
			if(strcmp(buffer, "/fy") == 0) in_predator_fy = 0;
			if(strcmp(buffer, "steer_x") == 0) in_predator_steer_x = 1;
			if(strcmp(buffer, "/steer_x") == 0) in_predator_steer_x = 0;
			if(strcmp(buffer, "steer_y") == 0) in_predator_steer_y = 1;
			if(strcmp(buffer, "/steer_y") == 0) in_predator_steer_y = 0;
			if(strcmp(buffer, "life") == 0) in_predator_life = 1;
			if(strcmp(buffer, "/life") == 0) in_predator_life = 0;
			if(strcmp(buffer, "id") == 0) in_grass_id = 1;
			if(strcmp(buffer, "/id") == 0) in_grass_id = 0;
			if(strcmp(buffer, "x") == 0) in_grass_x = 1;
			if(strcmp(buffer, "/x") == 0) in_grass_x = 0;
			if(strcmp(buffer, "y") == 0) in_grass_y = 1;
			if(strcmp(buffer, "/y") == 0) in_grass_y = 0;
			if(strcmp(buffer, "type") == 0) in_grass_type = 1;
			if(strcmp(buffer, "/type") == 0) in_grass_type = 0;
			if(strcmp(buffer, "dead_cycles") == 0) in_grass_dead_cycles = 1;
			if(strcmp(buffer, "/dead_cycles") == 0) in_grass_dead_cycles = 0;
			if(strcmp(buffer, "available") == 0) in_grass_available = 1;
			if(strcmp(buffer, "/available") == 0) in_grass_available = 0;
			
            /* environment variables */
            if(strcmp(buffer, "REPRODUCE_PREY_PROB") == 0) in_env_REPRODUCE_PREY_PROB = 1;
            if(strcmp(buffer, "/REPRODUCE_PREY_PROB") == 0) in_env_REPRODUCE_PREY_PROB = 0;
			if(strcmp(buffer, "REPRODUCE_PREDATOR_PROB") == 0) in_env_REPRODUCE_PREDATOR_PROB = 1;
            if(strcmp(buffer, "/REPRODUCE_PREDATOR_PROB") == 0) in_env_REPRODUCE_PREDATOR_PROB = 0;
			if(strcmp(buffer, "GAIN_FROM_FOOD_PREDATOR") == 0) in_env_GAIN_FROM_FOOD_PREDATOR = 1;
            if(strcmp(buffer, "/GAIN_FROM_FOOD_PREDATOR") == 0) in_env_GAIN_FROM_FOOD_PREDATOR = 0;
			if(strcmp(buffer, "GAIN_FROM_FOOD_PREY") == 0) in_env_GAIN_FROM_FOOD_PREY = 1;
            if(strcmp(buffer, "/GAIN_FROM_FOOD_PREY") == 0) in_env_GAIN_FROM_FOOD_PREY = 0;
			if(strcmp(buffer, "GRASS_REGROW_CYCLES") == 0) in_env_GRASS_REGROW_CYCLES = 1;
            if(strcmp(buffer, "/GRASS_REGROW_CYCLES") == 0) in_env_GRASS_REGROW_CYCLES = 0;
			
    
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
				if(in_prey_id){ 
                    prey_id = (int) atoi(buffer);    
                }
				if(in_prey_x){ 
                    prey_x = (float) atof(buffer);    
                }
				if(in_prey_y){ 
                    prey_y = (float) atof(buffer);    
                }
				if(in_prey_type){ 
                    prey_type = (float) atof(buffer);    
                }
				if(in_prey_fx){ 
                    prey_fx = (float) atof(buffer);    
                }
				if(in_prey_fy){ 
                    prey_fy = (float) atof(buffer);    
                }
				if(in_prey_steer_x){ 
                    prey_steer_x = (float) atof(buffer);    
                }
				if(in_prey_steer_y){ 
                    prey_steer_y = (float) atof(buffer);    
                }
				if(in_prey_life){ 
                    prey_life = (int) atoi(buffer);    
                }
				if(in_predator_id){ 
                    predator_id = (int) atoi(buffer);    
                }
				if(in_predator_x){ 
                    predator_x = (float) atof(buffer);    
                }
				if(in_predator_y){ 
                    predator_y = (float) atof(buffer);    
                }
				if(in_predator_type){ 
                    predator_type = (float) atof(buffer);    
                }
				if(in_predator_fx){ 
                    predator_fx = (float) atof(buffer);    
                }
				if(in_predator_fy){ 
                    predator_fy = (float) atof(buffer);    
                }
				if(in_predator_steer_x){ 
                    predator_steer_x = (float) atof(buffer);    
                }
				if(in_predator_steer_y){ 
                    predator_steer_y = (float) atof(buffer);    
                }
				if(in_predator_life){ 
                    predator_life = (int) atoi(buffer);    
                }
				if(in_grass_id){ 
                    grass_id = (int) atoi(buffer);    
                }
				if(in_grass_x){ 
                    grass_x = (float) atof(buffer);    
                }
				if(in_grass_y){ 
                    grass_y = (float) atof(buffer);    
                }
				if(in_grass_type){ 
                    grass_type = (float) atof(buffer);    
                }
				if(in_grass_dead_cycles){ 
                    grass_dead_cycles = (int) atoi(buffer);    
                }
				if(in_grass_available){ 
                    grass_available = (int) atoi(buffer);    
                }
				
			}
            else if (in_env){
                if(in_env_REPRODUCE_PREY_PROB){
                    //scalar value input
                    env_REPRODUCE_PREY_PROB = (float) atof(buffer);
                    set_REPRODUCE_PREY_PROB(&env_REPRODUCE_PREY_PROB);
                      
                }
                if(in_env_REPRODUCE_PREDATOR_PROB){
                    //scalar value input
                    env_REPRODUCE_PREDATOR_PROB = (float) atof(buffer);
                    set_REPRODUCE_PREDATOR_PROB(&env_REPRODUCE_PREDATOR_PROB);
                      
                }
                if(in_env_GAIN_FROM_FOOD_PREDATOR){
                    //scalar value input
                    env_GAIN_FROM_FOOD_PREDATOR = (int) atoi(buffer);
                    set_GAIN_FROM_FOOD_PREDATOR(&env_GAIN_FROM_FOOD_PREDATOR);
                      
                }
                if(in_env_GAIN_FROM_FOOD_PREY){
                    //scalar value input
                    env_GAIN_FROM_FOOD_PREY = (int) atoi(buffer);
                    set_GAIN_FROM_FOOD_PREY(&env_GAIN_FROM_FOOD_PREY);
                      
                }
                if(in_env_GRASS_REGROW_CYCLES){
                    //scalar value input
                    env_GRASS_REGROW_CYCLES = (int) atoi(buffer);
                    set_GRASS_REGROW_CYCLES(&env_GRASS_REGROW_CYCLES);
                      
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
	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

