/*
 * Copyright 2017 University of Sheffield.
 * Author: Peter Heywood
 * Contact: p.richmond@sheffield.ac.uk
 * (http://www.paulrichmond.staff.shef.ac.uk)
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
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <vector>

xmachine_memory_Person **h_agent_AoS;
xmachine_memory_Household **h_household_AoS;
const unsigned int h_agent_AoS_MAX = 32768;
const unsigned int h_household_AoS_MAX = 2048;
unsigned int h_nextID;
unsigned int h_nextHouseholdID;

__host__ unsigned int getNextID() {
  unsigned int old = h_nextID;
  h_nextID++;
  return old;
}

__host__ unsigned int getNextHouseholdID() {
  unsigned int old = h_nextHouseholdID;
  h_nextHouseholdID++;
  return old;
}

__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
  printf("Set TIME_STEP = %f\n", *get_TIME_STEP());
  printf("Set SCALE_FACTOR = %f\n", *get_SCALE_FACTOR());
  printf("Set MAX_AGE = %u\n", *get_MAX_AGE());
  printf("Set RANDOM_AGES = %u\n", *get_RANDOM_AGES());

  srand(0);

  h_nextID = 1;

  h_agent_AoS = h_allocate_agent_Person_array(h_agent_AoS_MAX);

  h_nextHouseholdID = 1;

  h_household_AoS = h_allocate_agent_Household_array(h_household_AoS_MAX);

  char const *const fileName = "data.in";
  FILE *file = fopen(fileName, "r");
  char line[256];

  unsigned int sizes = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int categories = strtol(fgets(line, sizeof(line), file), NULL, 0);

  unsigned int gender;
  unsigned int minage;

  char *maximumage;
  unsigned int maxage;

  unsigned int currentsize;
  unsigned int amount;
  unsigned int age;

  unsigned int sizearray[32];
  signed int count;

  for (unsigned int n = 0; n < 32; n++) {
    sizearray[n] = 0;
  }

  for (unsigned int i = 0; i < categories; i++) {

    gender = strtol(fgets(line, sizeof(line), file), NULL, 0);
    minage = strtol(fgets(line, sizeof(line), file), NULL, 0);

    maximumage = fgets(line, sizeof(line), file);

    if (not strcmp(maximumage, "Inf\n")) {
      maxage = (*get_MAX_AGE());
    } else {
      maxage = strtol(maximumage, NULL, 0);
    }

    for (unsigned int ii = 0; ii < sizes; ii++) {

      currentsize = strtol(fgets(line, sizeof(line), file), NULL, 0);
      amount = strtol(fgets(line, sizeof(line), file), NULL, 0);

      for (unsigned int iii = 0; iii < amount; iii++) {

        xmachine_memory_Person *h_person = h_allocate_agent_Person();

        age = (rand() % maxage) + minage;

        h_person->id = getNextID();
        h_person->age = age;
        h_person->gender = gender;
        h_person->householdsize = currentsize;

        sizearray[currentsize]++;

        h_add_agent_Person_default(h_person);

        h_free_agent_Person(&h_person);
      }
    }
  }

  count = 0;

  for (unsigned int h = 1; h < 32; h++) {

    for (unsigned int hh = 0; hh < (sizearray[h] / h); hh++) {
      xmachine_memory_Household *h_household = h_allocate_agent_Household();

      h_household->id = getNextHouseholdID();
      h_household->size = h;

      for (unsigned int hhh = 0; hhh < h; hhh++) {
        h_household->people[hhh] = count;
        count++;
      }

      h_add_agent_Household_hhdefault(h_household);

      h_free_agent_Household(&h_household);
    }
  }

  while (fgets(line, sizeof(line), file)) {
    printf("%s", line);
  }

  printf("Sizes = %u\n", sizes);
  printf("Categories = %u\n", categories);

  fclose(file);
}

__FLAME_GPU_INIT_FUNC__ void generateAgentsInit() {

  /* printf("Population after init function: %u\n",
          get_agent_Person_default_count());

   h_nextHouseholdID = 1;
   h_household_AoS = h_allocate_agent_Household_array(h_household_AoS_MAX);

   unsigned int sizearray[h_agent_AoS_MAX][32];
   unsigned int currentsize;
   unsigned int count;

   for (int i = 0; i < 32; i++) {
     sizearray[0][i] = 0;
   }

   for (int index = 0; index < get_agent_Person_default_count(); index++) {
     currentsize = get_Person_default_variable_householdsize(index);
     count = sizearray[0][currentsize];
     sizearray[count][currentsize] = get_Person_default_variable_id(index);
     sizearray[0][currentsize]++;
   }

   printf("Here at least?");

   for (int i = 0; i < 32; i++) {
     printf("Made it!");
     count = sizearray[0][i];
     sizearray[count][i] = INT_MAX;

     count = 1;

     while (sizearray[count][i] != INT_MAX) {

       xmachine_memory_Household *h_household = h_allocate_agent_Household();

       h_household->id = getNextHouseholdID();
       h_household->size = i;

       for (int ii = 0; ii < i; ii++) {
         if (sizearray[count][i] != INT_MAX) {
           h_household->people[ii] = sizearray[count][i];
           count++;
           printf("Add");
         } else {
           printf("Don't");
         }
       }

       h_add_agent_Household_hhdefault(h_household);

       h_free_agent_Household(&h_household);
     }
   }*/
}

__FLAME_GPU_STEP_FUNC__ void generatePersonStep() {

  printf("Population after step function %u\n",
         get_agent_Person_default_count());
}

__FLAME_GPU_STEP_FUNC__ void customOutputStepFunction() {

  const char *directory = getOutputDir();
  unsigned int iteration = getIterationNumber();

  if (iteration % 5 == 0) {

    std::string outputFilename =
        std::string(std::string(directory) + "person-output-" +
                    std::to_string(iteration) + ".csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Person data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, gender, age\n");

      for (int index = 0; index < get_agent_Person_default_count(); index++) {

        fprintf(fp, "%u, %u, %u\n", get_Person_default_variable_id(index),
                get_Person_default_variable_gender(index),
                get_Person_default_variable_age(index));
      }

      fflush(fp);
    } else {
      fprintf(
          stderr,
          "Error: file %s could not be created for customOutputStepFunction\n",
          outputFilename.c_str());
    }

    if (fp != nullptr && fp != stdout && fp != stderr) {
      fclose(fp);
      fp = nullptr;
    }
  }

  if (iteration == 1) {

    std::string outputFilename =
        std::string(std::string(directory) + "household-output.csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Household data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, size\n");

      for (int index = 0; index < get_agent_Household_hhdefault_count();
           index++) {

        fprintf(fp, "%u, %u\n", get_Household_hhdefault_variable_id(index),
                get_Household_hhdefault_variable_size(index));
      }

      fflush(fp);
    } else {
      fprintf(
          stderr,
          "Error: file %s could not be created for customOutputStepFunction\n",
          outputFilename.c_str());
    }

    if (fp != nullptr && fp != stdout && fp != stderr) {
      fclose(fp);
      fp = nullptr;
    }
  }
}

__FLAME_GPU_EXIT_FUNC__ void exitFunction() {

  h_free_agent_Person_array(&h_agent_AoS, h_agent_AoS_MAX);

  printf("Population for exit function: %u\n",
         get_agent_Person_default_count());
}

__FLAME_GPU_FUNC__ int update(xmachine_memory_Person *agent,
                              RNG_rand48 *rand48) {

  agent->age += TIME_STEP;
  float random = rnd<CONTINUOUS>(rand48);

  if (random < (agent->age * SCALE_FACTOR * TIME_STEP)) {
    return 1;
  }
  return 0;
}

__FLAME_GPU_FUNC__ int hhupdate(xmachine_memory_Household *agent) { return 0; }

#endif
