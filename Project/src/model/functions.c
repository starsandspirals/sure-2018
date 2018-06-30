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
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

xmachine_memory_Person **h_agent_AoS;
xmachine_memory_Household **h_household_AoS;
const unsigned int h_agent_AoS_MAX = 32768;
const unsigned int h_household_AoS_MAX = 8192;
const float beta0 = 2.19261;
const float beta1 = 0.14679;
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

__host__ void shuffle(unsigned int *array1, unsigned int *array2, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned int t1 = array1[j];
      unsigned int t2 = array2[j];
      array1[j] = array1[i];
      array2[j] = array2[i];
      array1[i] = t1;
      array2[i] = t2;
    }
  }
}

__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
  printf("Set TIME_STEP = %f\n", *get_TIME_STEP());
  printf("Set SCALE_FACTOR = %f\n", *get_SCALE_FACTOR());
  printf("Set MAX_AGE = %u\n", *get_MAX_AGE());
  printf("Set RANDOM_AGES = %u\n", *get_RANDOM_AGES());
  printf("Set STARTING_POPULATION = %u\n", (int)*get_STARTING_POPULATION());

  srand(0);

  h_nextID = 1;

  h_agent_AoS = h_allocate_agent_Person_array(h_agent_AoS_MAX);

  h_nextHouseholdID = 1;

  h_household_AoS = h_allocate_agent_Household_array(h_household_AoS_MAX);

  char const *const fileName = "data.in";
  FILE *file = fopen(fileName, "r");
  char line[256];

  unsigned int total = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int sizes = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int categories = strtol(fgets(line, sizeof(line), file), NULL, 0);

  unsigned int gender;
  unsigned int minage;

  char *maximumage;
  unsigned int maxage;

  unsigned int currentsize;
  float amount;
  unsigned int rounded;
  unsigned int age;

  unsigned int sizearray[32];
  signed int count;
  unsigned int ages[h_agent_AoS_MAX];

  for (unsigned int i = 0; i < 32; i++) {
    sizearray[i] = 0;
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
      amount = strtof(fgets(line, sizeof(line), file), NULL);

      rounded = round((amount / total) * *get_STARTING_POPULATION());

      for (unsigned int iii = 0; iii < rounded; iii++) {

        xmachine_memory_Person *h_person = h_allocate_agent_Person();

        age = (rand() % (maxage - minage)) + minage;

        h_person->id = getNextID();
        h_person->age = age;
        h_person->gender = gender;
        h_person->householdsize = currentsize;

        sizearray[currentsize]++;
        ages[h_person->id] = age;

        h_add_agent_Person_default(h_person);

        h_free_agent_Person(&h_person);
      }
    }
  }

  count = 0;
  float churchprob;
  total = get_agent_Person_default_count();
  unsigned int order[total];

  for (unsigned int i = 0; i < total; i++) {
    order[i] = i;
  }

  shuffle(order, ages, total);

  unsigned int adult[h_household_AoS_MAX];
  unsigned int adultcount;

  for (unsigned int i = 1; i < 32; i++) {
    for (unsigned int ii = 0; ii < (sizearray[i] / i); ii++) {
      xmachine_memory_Household *h_household = h_allocate_agent_Household();
      churchprob = 1 / (1 + exp(-beta0 - (beta1 * i)));
      adultcount = 0;

      h_household->id = getNextHouseholdID();
      h_household->size = i;

      for (unsigned int iii = 0; iii < i; iii++) {
        h_household->people[iii] = order[count];
        count++;

        if (ages[count] >= 15) {
          adultcount++;
        }
      }

      float random = ((float)rand() / (RAND_MAX));

      if (random < churchprob) {
        h_household->churchgoing = 1;
      } else {
        h_household->churchgoing = 0;
      }

      if (h_household->churchgoing) {
        random = ((float)rand() / (RAND_MAX));
        if (random < 0.285569106) {
          h_household->churchfreq = 0;
        } else if (random < 0.704268293) {
          h_household->churchfreq = 1;
        } else if (random < 0.864329269) {
          h_household->churchfreq = 2;
        } else if (random < 0.944613822) {
          h_household->churchfreq = 3;
        } else if (random < 0.978658537) {
          h_household->churchfreq = 4;
        } else if (random < 0.981707317) {
          h_household->churchfreq = 5;
        } else if (random < 0.985772358) {
          h_household->churchfreq = 6;
        } else {
          h_household->churchfreq = 7;
        }
      } else {
        h_household->churchfreq = 0;
      }

      h_household->adults = adultcount;
      adult[h_household->id] = adultcount;

      h_add_agent_Household_hhdefault(h_household);

      h_free_agent_Household(&h_household);
    }
  }

  unsigned int hhtotal = get_agent_Household_hhdefault_count();
  unsigned int hhorder[hhtotal];

  for (unsigned int i = 0; i < hhtotal; i++) {
    hhorder[i] = i;
  }

  shuffle(hhorder, adult, hhtotal);

  while (fgets(line, sizeof(line), file)) {
    printf("%s", line);
  }

  printf("Sizes = %u\n", sizes);
  printf("Categories = %u\n", categories);

  fclose(file);
}

__FLAME_GPU_INIT_FUNC__ void generateAgentsInit() {

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

      fprintf(fp, "ID, size, churchgoing, churchfreq\n");

      for (int index = 0; index < get_agent_Household_hhdefault_count();
           index++) {

        fprintf(fp, "%u, %u, %u, %u\n",
                get_Household_hhdefault_variable_id(index),
                get_Household_hhdefault_variable_size(index),
                get_Household_hhdefault_variable_churchgoing(index),
                get_Household_hhdefault_variable_churchfreq(index));
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

__FLAME_GPU_FUNC__ int chuupdate(xmachine_memory_Church *agent) { return 0; }

#endif
