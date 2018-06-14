/*
 * Copyright 2017 University of Sheffield.
 * Authors: Dr Mozhgan Kabiri chimeh, Dr Paul Richmond
 * Contact: m.kabiri-chimeh@sheffield.ac.uk (http://www.mkchimeh.staff.shef.ac.uk)
 p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
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

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"

//helpful definitions
#define float2 glm::vec2
#define length(x) glm::length(x)
#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x


//Environment Bounds (should match the bounds in the 0.xml generator)
#define MIN_POSITION -1.0f
#define MAX_POSITION +1.0f
#define BOUNDS_WIDTH (MAX_POSITION - MIN_POSITION)



//Hard coded model parameters
#define PRED_PREY_INTERACTION_RADIUS			0.100f			//The radius in which predators chase prey and prey run away from predators		
#define PREY_GROUP_COHESION_RADIUS				0.200f			//The radius in which prey agents form cohesive groups
#define SAME_SPECIES_AVOIDANCE_RADIUS			0.035f			//The radius in which agents of the same species will avoid each other (close proximity collision avoidance behaviour)

#define GRASS_EAT_DISTANCE						0.020f			//The distance in which grass is eaten by prey
#define PRED_KILL_DISTANCE						0.020f			//The distance in which prey are eaten by predators

#define DELTA_TIME								0.001f			//Time step integral
#define PRED_SPEED_ADVANTAGE					2.000f			//Speed multiplier to give predators an advantage in catching prey


//Global file variable for logging
FILE *count_output;


/*---------------------------------------- GPU helper functions ----------------------------------------*/

/* Environment forms a continuous torus */
__FLAME_GPU_FUNC__ float2 boundPosition(float2 agent_position)
{
	agent_position.x = (agent_position.x < MIN_POSITION) ? MAX_POSITION : agent_position.x;
	agent_position.x = (agent_position.x > MAX_POSITION) ? MIN_POSITION : agent_position.x;

	agent_position.y = (agent_position.y < MIN_POSITION) ? MAX_POSITION : agent_position.y;
	agent_position.y = (agent_position.y > MAX_POSITION) ? MIN_POSITION : agent_position.y;

	return agent_position;
}



/*---------------------------------------- INIT, STEP and EXIT functions ----------------------------------------*/

/* This is a CPU function which is listed in the models stepFunctions XMML element. It is called after each simulation iteration. */
__FLAME_GPU_STEP_FUNC__ void outputToLogFile(){
	//Get the agent population counts
	int predator = get_agent_predator_default2_count();
	int prey = get_agent_prey_default1_count();
	//Use a FLAME GPU parallel reduction function to get all grass which is available
	int grass_variable = reduce_grass_default3_available_variable();

	//Output population and quantity counts
	fprintf(count_output, "Prey ,%d, Predator ,%d, Grass ,%d\n", prey, predator, grass_variable);
}

/* This is a CPU function which is listed in the models initFunctions XMML element. It is called after the initial states are loaded but before the first iteration of the simulation. */
__FLAME_GPU_INIT_FUNC__ void initLogFile(){
	//Open a file for logging during simulation
	char output_file[1024];
	sprintf(output_file, "%s%s", getOutputDir(), "PreyPred_Count.csv");
	count_output = fopen(output_file, "w");

	//output initial count values at time=0
	outputToLogFile();
}

/* This is a CPU function which is listed in the models exitFunctions XMML element. It is called afterthe final simulation iteration. */
__FLAME_GPU_EXIT_FUNC__ void closeLogFile(){
	//Close the log file at the end of the simulation
	fclose(count_output);
}

/*---------------------------------------- PREY FUNCTIONS ----------------------------------------*/

__FLAME_GPU_FUNC__ int prey_output_location(xmachine_memory_prey* xmemory, xmachine_message_prey_location_list* prey_location_messages)
{
	//Add a prey location message
	add_prey_location_message(prey_location_messages, xmemory->id, xmemory->x, xmemory->y);
	return 0;
}



// Preys avoids predators
__FLAME_GPU_FUNC__ int prey_avoid_pred(xmachine_memory_prey* agent, xmachine_message_pred_location_list* pred_location_messages){

	float2 agent_position = float2(agent->x, agent->y);
	float2 agent_velocity = float2(agent->fx, agent->fy);
	float2 average_center = float2(0.0f, 0.0f);
	float2 avoid_velocity = float2(0.0f, 0.0f);

	//Iterate the predator location messages until NULL is returned which indicates all messages have been read.
	xmachine_message_pred_location* current_message = get_first_pred_location_message(pred_location_messages);
	while (current_message)
	{

		float2 message_position = float2(current_message->x, current_message->y);
		float separation = length(agent_position - message_position);

		//Prey have a avoidance velocity which increases as the distance between them decreases
		if (separation < (PRED_PREY_INTERACTION_RADIUS))
		{
			if (separation > 0.0f)
				avoid_velocity += PRED_PREY_INTERACTION_RADIUS / separation*(agent_position - message_position);

		}
		
		current_message = get_next_pred_location_message(current_message, pred_location_messages);
	}

	//Set the steering force to the avoidance force
	agent->steer_x = avoid_velocity.x;
	agent->steer_y = avoid_velocity.y;

	return 0;
}

// update preys location
__FLAME_GPU_FUNC__ int prey_flock(xmachine_memory_prey* xmemory, xmachine_message_prey_location_list* prey_location_messages)
{
	//Agent position vector
	float2 agent_position = float2(xmemory->x, xmemory->y);


	float2 group_center = float2(0.0f, 0.0f);
	float2 group_velocity = float2(0.0f, 0.0f);
	float2 avoid_velocity = float2(0.0f, 0.0f);
	int group_centre_count = 0;

	//Iterate the prey location messages until NULL is returned which indicates all messages have been read.
	xmachine_message_prey_location* prey_location_message = get_first_prey_location_message(prey_location_messages);
	while (prey_location_message)
	{
		float2 message_position = float2(prey_location_message->x, prey_location_message->y);

		float separation = length(agent_position - message_position);
		if (separation < (PREY_GROUP_COHESION_RADIUS) && (prey_location_message->id != xmemory->id))
		{

			//Update Perceived global centre for grouping velocity
			group_center += message_position;
			group_centre_count += 1;

			//calculate a avoidance velocity which increases as the distance between prey decreases
			if (separation < (SAME_SPECIES_AVOIDANCE_RADIUS))
			{
				if (separation > 0.0f)
					avoid_velocity += SAME_SPECIES_AVOIDANCE_RADIUS/separation*(agent_position - message_position);
			}

		}

		prey_location_message = get_next_prey_location_message(prey_location_message, prey_location_messages);
	}

	//Average nearby agents positions to find the centre of the group and create a velocity to move towards it 
	if (group_centre_count){
		group_center /= group_centre_count;
		group_velocity = (group_center - agent_position);
	}

	//Add the grouping and avoidance velocities to the steer velocity (which already contains the predator avoidance velocity)
	xmemory->steer_x += group_velocity.x + avoid_velocity.x;
	xmemory->steer_y += group_velocity.y + avoid_velocity.y;
	
	
	return 0;
}


// update preys location
__FLAME_GPU_FUNC__ int prey_move(xmachine_memory_prey* xmemory)
{
	
	//Agent position vector
	float2 agent_position = float2(xmemory->x, xmemory->y);
	float2 agent_velocity = float2(xmemory->fx, xmemory->fy);
	float2 agent_steer = float2(xmemory->steer_x, xmemory->steer_y);
	
	//Adjust the velocity according to the steering velocity
	agent_velocity += agent_steer;

	//Limit the speed of the avoidance velocity
	float current_speed = length(agent_velocity);
	if (current_speed > 1.0f){
		agent_velocity = normalize(agent_velocity);
	}

	//Integrate the position by applying moving according to the velocity
	agent_position += agent_velocity * DELTA_TIME;


	//Bound the position within the environment 
	agent_position = boundPosition(agent_position);

	//Update the agents position and velocity
	xmemory->x = agent_position.x;
	xmemory->y = agent_position.y;
	xmemory->fx = agent_velocity.x;
	xmemory->fy = agent_velocity.y;

	//reduce life by one unit of energy
	xmemory->life--;

	return 0;
}


__FLAME_GPU_FUNC__ int prey_eaten(xmachine_memory_prey* xmemory, xmachine_message_pred_location_list* pred_location_messages, xmachine_message_prey_eaten_list* prey_eaten_messages)
{
	int eaten = 0;
	int predator_id = -1;
	float closest_pred = PRED_KILL_DISTANCE;

	//Iterate the predator location messages until NULL is returned which indicates all messages have been read.
	xmachine_message_pred_location* pred_location_message = get_first_pred_location_message(pred_location_messages);
	while (pred_location_message)
	{
		//calculate distance between prey and predator
		float2 predator_pos = float2(pred_location_message->x, pred_location_message->y);
		float2 prey_pos = float2(xmemory->x, xmemory->y);
		float distance = length(predator_pos - prey_pos);

		//if distance is closer than nearest predator so far then select this predator as the one which will eat the prey
		if (distance < closest_pred)
		{
			predator_id = pred_location_message->id;
			closest_pred = distance;
			eaten = 1;
		}

		pred_location_message = get_next_pred_location_message(pred_location_message, pred_location_messages);
	}

	//if one or more predators were within killing distance then notify the nearest predator that it has eaten this prey via a prey eaten message.
	if (eaten)
		add_prey_eaten_message(prey_eaten_messages, predator_id);

	//return eaten value to remove dead (eaten == 1) agents from the simulation
	return eaten;
}

__FLAME_GPU_FUNC__ int prey_eat_or_starve(xmachine_memory_prey* xmemory, xmachine_message_grass_eaten_list* grass_eaten_messages)
{
	int dead = 0;

        // Excercise 3.3

	//return dead value to remove dead agents from the simulation
	return dead;
}

// generate prey
__FLAME_GPU_FUNC__ int prey_reproduction(xmachine_memory_prey* agent_prey, xmachine_memory_prey_list* agent_prey_agents, RNG_rand48* rand48)
{
	//generate a random number <CONTINUOUS> for agent in continuous space
	float random = rnd<CONTINUOUS>(rand48);
	if (random < REPRODUCE_PREY_PROB)
	{
		//Generate new random locations and velocities. 
		//The new id is a large random number with low probability of collisions.
		int id = (int)(rnd<CONTINUOUS>(rand48)*(float)INT_MAX);
		float x = rnd<CONTINUOUS>(rand48)*BOUNDS_WIDTH - BOUNDS_WIDTH/2.0f;
		float y = rnd<CONTINUOUS>(rand48)*BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
		float fx = rnd<CONTINUOUS>(rand48) * 2 - 1;
		float fy = rnd<CONTINUOUS>(rand48) * 2 - 1;

		//Parent and child shares the parent energy
		agent_prey->life /= 2;

		//add the new agent to the simulation
		add_prey_agent(agent_prey_agents, id, x, y, 1.0f, fx, fy, 0.0f, 0.0f, agent_prey->life);

	}

	return 0;
}



/*---------------------------------------- PREDATOR FUNCTIONS ----------------------------------------*/

__FLAME_GPU_FUNC__ int pred_output_location(xmachine_memory_predator* xmemory, xmachine_message_pred_location_list* pred_location_messages)
{
	add_pred_location_message(pred_location_messages, xmemory->id, xmemory->x, xmemory->y);
	return 0;
}



// Preds move towards preys
__FLAME_GPU_FUNC__ int pred_follow_prey(xmachine_memory_predator* xmemory, xmachine_message_prey_location_list* prey_location_messages){

	float2 agent_position = float2(xmemory->x, xmemory->y);
	float2 agent_steer = float2(0.0f, 0.0f);

	float2 closest_prey_position = float2(0.0f, 0.0f);
	float closest_prey_distance = PRED_PREY_INTERACTION_RADIUS;
	int can_see_prey = 0;

	//Iterate the prey location messages until NULL is returned which indicates all messages have been read.
	xmachine_message_prey_location* current_message = get_first_prey_location_message(prey_location_messages);
	while (current_message)
	{

		float2 message_position = float2(current_message->x, current_message->y);
		float separation = length(agent_position - message_position);

		//if distacne between predator and prey location is closest encountered so far then record the position and distance
		if ((separation < closest_prey_distance)){
			closest_prey_position = message_position;
			closest_prey_distance = separation;
			can_see_prey = 1;
		}

		current_message = get_next_prey_location_message(current_message, prey_location_messages);
	}

	//if there was a prey visible then create a velocity vector which will move the predator towards it.
	if (can_see_prey)
		agent_steer = closest_prey_position - agent_position;

	//set the steering vector
	xmemory->steer_x = agent_steer.x;
	xmemory->steer_y = agent_steer.y;

	return 0;
}



__FLAME_GPU_FUNC__ int pred_avoid(xmachine_memory_predator* xmemory, xmachine_message_pred_location_list* pred_location_messages)
{
	float2 agent_position = float2(xmemory->x, xmemory->y);
	float2 avoid_velocity = float2(0.0f, 0.0f);

	//Iterate the predator location messages until NULL is returned which indicates all messages have been read.
	xmachine_message_pred_location* pred_location_message = get_first_pred_location_message(pred_location_messages);
	while (pred_location_message)
	{
		float2 message_position = float2(pred_location_message->x, pred_location_message->y);
		float separation = length(agent_position - message_position);

		//Predators avoid each other with a force which increases as the distance between them decreases
		if (separation < (SAME_SPECIES_AVOIDANCE_RADIUS) && (pred_location_message->id != xmemory->id))
		{
			if (separation > 0.0f)
				avoid_velocity += SAME_SPECIES_AVOIDANCE_RADIUS / separation*(agent_position - message_position);
		}

		pred_location_message = get_next_pred_location_message(pred_location_message, pred_location_messages);
	}

	//Update the steering velocity which already has a velocity for chasing the nearest prey
	xmemory->steer_x += avoid_velocity.x;
	xmemory->steer_y += avoid_velocity.y;

	return 0;
}



// update preds locations
__FLAME_GPU_FUNC__ int pred_move(xmachine_memory_predator* xmemory)
{
	float2 agent_position = float2(xmemory->x, xmemory->y);
	float2 agent_velocity = float2(xmemory->fx, xmemory->fy);
	float2 agent_steer = float2(xmemory->steer_x, xmemory->steer_y);

	//Adjust the velocity according to the steering velocity
	agent_velocity += agent_steer;

	//Limit the speed of the velocity
	float current_speed = length(agent_velocity);
	if (current_speed > 1.0f){
		agent_velocity = normalize(agent_velocity);
	}

	//Integrate the position by applying moving according to the velocity. Predators can move faster than prey by some factor.
	agent_position += agent_velocity * DELTA_TIME * PRED_SPEED_ADVANTAGE;

	//Bound the position within the environment 
	agent_position = boundPosition(agent_position);

	//Update the agents position and velocity
	xmemory->x = agent_position.x;
	xmemory->y = agent_position.y;
	xmemory->fx = agent_velocity.x;
	xmemory->fy = agent_velocity.y;

	//reduce life by one unit of energy
	xmemory->life--;

	return 0;
}


__FLAME_GPU_FUNC__ int pred_eat_or_starve(xmachine_memory_predator* xmemory, xmachine_message_prey_eaten_list* prey_eaten_messages)
{
	int dead = 0;

	//Iterate the prey eaten messages until NULL is returned which indicates all messages have been read.
	xmachine_message_prey_eaten* prey_eaten_message = get_first_prey_eaten_message(prey_eaten_messages);
	while (prey_eaten_message)
	{
		//if the prey eaten message indicates that this predator ate some prey then increase the predators life by adding energy
		if (xmemory->id == prey_eaten_message->pred_id) {
			xmemory->life += GAIN_FROM_FOOD_PREDATOR;
		}

		prey_eaten_message = get_next_prey_eaten_message(prey_eaten_message, prey_eaten_messages);
	}

	//if the life has reduced to 0 then the prey should die or starvation
	if (xmemory->life < 1)
	{
		dead = 1;
	}

	//return dead value to remove dead agents from the simulation
	return dead;
}



// generate predator
__FLAME_GPU_FUNC__ int pred_reproduction(xmachine_memory_predator* agent_predator, xmachine_memory_predator_list* agent_predator_agents, RNG_rand48* rand48)
{
	//Generate a random number <CONTINUOUS> for agent in continuous space
	//Compare random number with probability of reproduction
	float random = rnd<CONTINUOUS>(rand48);
	if (random < REPRODUCE_PREDATOR_PROB)
	{
		//Generate new random locations and velocities. 
		//The new id is a large random number with low probability of collisions.
		int id = (int)(rnd<CONTINUOUS>(rand48)*(float)INT_MAX);
		float x = rnd<CONTINUOUS>(rand48)*BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
		float y = rnd<CONTINUOUS>(rand48)*BOUNDS_WIDTH - BOUNDS_WIDTH / 2.0f;
		float fx = rnd<CONTINUOUS>(rand48) * 2 - 1;
		float fy = rnd<CONTINUOUS>(rand48) * 2 - 1;

		//Parent and child shares the parents energy
		agent_predator->life /= 2;

		//add the new agent to the simulation
		add_predator_agent(agent_predator_agents, id, x, y, 0.0f, fx, fy, 0.0f, 0.0f, agent_predator->life);

	}

	return 0;
}


/*---------------------------------------- GRASS FUNCTIONS ----------------------------------------*/


__FLAME_GPU_FUNC__ int grass_output_location(xmachine_memory_grass* xmemory, xmachine_message_grass_location_list* grass_location_messages)
{
	// Excercise 3.1 : add location message
	add_grass_location_message(grass_location_messages, xmemory->id, xmemory->x, xmemory->y);
	return 0;
}




__FLAME_GPU_FUNC__ int grass_eaten(xmachine_memory_grass* xmemory, xmachine_message_prey_location_list* prey_location_messages, xmachine_message_grass_eaten_list* grass_eaten_messages)
{
	// Excercise 3.2 

	return 0;
}

// generate grass
__FLAME_GPU_FUNC__ int grass_growth(xmachine_memory_grass* agent_grass, RNG_rand48* rand48)
{
        // Excercise 3.4
	
	return 0;
}

#endif // #ifndef _FUNCTIONS_H_


