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
#include <stdio.h>
#include <string.h>
#include <vector>

xmachine_memory_Person **h_agent_AoS;
const unsigned int h_agent_AoS_MAX = 32768;
unsigned int h_nextID;

__host__ unsigned int getNextID() {
  unsigned int old = h_nextID;
  h_nextID++;
  return old;
}

__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
  printf("Set TIME_STEP = %f\n", *get_TIME_STEP());
  printf("Set SCALE_FACTOR = %f\n", *get_SCALE_FACTOR());
  printf("Set MAX_AGE = %u\n", *get_MAX_AGE());
  printf("Set RANDOM_AGES = %u\n", *get_RANDOM_AGES());

  srand(0);

  h_nextID = 0;

  h_agent_AoS = h_allocate_agent_Person_array(h_agent_AoS_MAX);

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

        h_add_agent_Person_default(h_person);

        h_free_agent_Person(&h_person);
      }
    }
  }

  while (fgets(line, sizeof(line), file)) {
    printf("%s", line);
  }

  printf("Sizes = %u\n", sizes);
  printf("Categories = %u\n", categories);

  fclose(file);
}

__FLAME_GPU_INIT_FUNC__ void generatePersonInit() {

  printf("Population after init function: %u\n",
         get_agent_Person_default_count());
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
        std::string(std::string(directory) + "custom-output-" +
                    std::to_string(iteration) + ".csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Person data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, age\n");

      for (int index = 0; index < get_agent_Person_default_count(); index++) {

        fprintf(fp, "%u, %u\n", get_Person_default_variable_id(index),
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
