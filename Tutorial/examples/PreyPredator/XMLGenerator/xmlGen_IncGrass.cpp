/*
This programs geneates the .xml input file.
Author : Mozhgan K. Chimeh
To Compile: g++ -std=gnu++11 xmlGen_IncGrass.cpp -o xmlGen
To Execute: ./xmlGen ../iterations/0.xml 800 400 2000 0.05 0.03 75 50 100
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>

#ifdef _WIN32
#define srand48(x) srand(x)
#define drand48() ((double)rand()/RAND_MAX)
#endif

#ifndef PI
#define PI						3.142857143f
#endif


int main( int argc, char** argv)
{
     float prey_rate;
    float predator_rate ;
     float grass_rate ;
     int energy_prey ;
    int energy_pred ;
    int grass_regrow ;
    
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(123); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1, 1); // positions, and velocity
    std::uniform_real_distribution<> dis3(0, 40); // pred life
    std::uniform_real_distribution<> dis4(0, 50); // prey life

	//proper error checking
	if (!(argc == 5 || argc == 10)){ 
		printf("Incorrect use of program. Format should be:\n");
		printf("  xmlGen 0.xml prey_num pred_num grass_num \nOR \n");
		printf("  xmlGen 0.xml prey_num pred_num grass_num prey_rate pred_rate energy_prey energy_pred grass_regrow\n");
		printf("\twhere\n");
		printf("\t0.xml is the Flame GPU simulation state xml file\n");
		return 1;
	}

    int prey_num = atoi(argv[2]);
    int pred_num = atoi(argv[3]);
    int grass_num = atoi(argv[4]);
    
 if (argc == 10){
    prey_rate = atof(argv[5]);
     predator_rate = atof(argv[6]);
      energy_prey = atoi(argv[7]);
     energy_pred = atoi(argv[8]);
     grass_regrow = atoi(argv[9]);
    }
    
    char * fileName = argv[1];
    int n = 0;
    int i = 0;

    FILE *fp = fopen(fileName, "w"); // write only

    // test for files not existing.
    if (fp== NULL) {
        printf("Error! Could not open file\n");
        exit(-1); // must include stdlib.h
    }

    fprintf(fp, "<states>\n");
    fprintf(fp, "<itno>0</itno>\n");
    fprintf(fp, "<environment>\n"); 
  if (argc == 10){
    fprintf(fp,"<REPRODUCE_PREY_PROB>%f</REPRODUCE_PREY_PROB>\n",prey_rate);
    fprintf(fp,"<REPRODUCE_PREDATOR_PROB>%f</REPRODUCE_PREDATOR_PROB>\n",predator_rate);
    fprintf(fp,"<GAIN_FROM_FOOD_PREDATOR>%d</GAIN_FROM_FOOD_PREDATOR>\n",energy_pred);
    fprintf(fp,"<GAIN_FROM_FOOD_PREY>%d</GAIN_FROM_FOOD_PREY>\n",energy_prey);
    fprintf(fp,"<GRASS_REGROW_CYCLES>%d</GRASS_REGROW_CYCLES>\n",grass_regrow);
    }
    fprintf(fp,"</environment>\n");
    
    while (n < grass_num) {

	float x = dis(gen) ;//* environment_width;
	float y = dis(gen) ;//* environment_width;
	
	fprintf(fp, "<xagent>\n");
	fprintf(fp, "<name>grass</name>\n");

        fprintf(fp, "<id>%d</id> \n",++i);
        fprintf(fp, "<x>%f</x>\n", x);
        fprintf(fp, "<y>%f</y>\n", y);
        fprintf(fp, "<type>2.0f</type>\n");
        fprintf(fp, "<dead_cycles>0</dead_cycles>\n");
        fprintf(fp, "<available>1</available>\n");
        fprintf(fp, "</xagent>\n");
        n++;
        
    }
    n = 0;
    i = 0;    
    while (n < prey_num) {

	float x = dis(gen) ;
	float y = dis(gen) ;
	
	float fx = dis(gen);
	float fy = dis(gen);
	
	int life = dis4(gen);
	
	fprintf(fp, "<xagent>\n");
	fprintf(fp, "<name>prey</name>\n");

        fprintf(fp, "<id>%d</id> \n",++i);
        fprintf(fp, "<x>%f</x>\n", x);
        fprintf(fp, "<y>%f</y>\n", y);
        fprintf(fp, "<type>1.0f</type>\n");
        fprintf(fp, "<fx>%f</fx>\n",fx);
        fprintf(fp, "<fy>%f</fy>\n",fy);
        fprintf(fp, "<steer_x>0.0f</steer_x>\n");
        fprintf(fp, "<steer_y>0.0f</steer_y>\n");
        fprintf(fp, "<life>%d</life>\n",life);
        fprintf(fp, "</xagent>\n");
        n++;
        
    }
    n = 0;
    i = 0;
    while (n < pred_num) {

	float x = dis(gen) ;
	float y = dis(gen) ;
	
	float fx = dis(gen);
	float fy = dis(gen);
	
        float speed = dis(gen);
	float steer_x = dis(gen);
	float steer_y = dis(gen);
	
	int life = dis3(gen);
	
        fprintf(fp, "<xagent>\n");
	fprintf(fp, "<name>predator</name>\n");
	
        fprintf(fp, "<id>%d</id> \n",++i);
        fprintf(fp, "<x>%f</x>\n", x);
        fprintf(fp, "<y>%f</y>\n", y);
        fprintf(fp, "<type>0.0f</type>\n");
        fprintf(fp, "<fx>%f</fx>\n",fx);
        fprintf(fp, "<fy>%f</fy>\n",fy);
        fprintf(fp, "<steer_x>0.0f</steer_x>\n");
        fprintf(fp, "<steer_y>0.0f</steer_y>\n");
        fprintf(fp, "<life>%d</life>\n",life);
        fprintf(fp, "</xagent>\n");
        n++;
        
    }
    fprintf(fp, "</states>");
    fclose(fp);
    return 0;
}
