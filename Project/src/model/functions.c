#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

// Allocate blocks of memory for each type of agent, and defining a constant
// for the maximum number of each that should be generated.
xmachine_memory_Person **h_agent_AoS;
xmachine_memory_Household **h_household_AoS;
xmachine_memory_Church **h_church_AoS;
xmachine_memory_Transport **h_transport_AoS;
const unsigned int h_agent_AoS_MAX = 32768;
const unsigned int h_household_AoS_MAX = 8192;
const unsigned int h_church_AoS_MAX = 256;
const unsigned int h_transport_AoS_MAX = 2048;

// Create variables for the next unused ID for each agent type, so that they
// remain unique, and also get functions to update the ID each time.
unsigned int h_nextID;
unsigned int h_nextHouseholdID;
unsigned int h_nextChurchID;
unsigned int h_nextTransportID;

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

__host__ unsigned int getNextChurchID() {
  unsigned int old = h_nextChurchID;
  h_nextChurchID++;
  return old;
}

__host__ unsigned int getNextTransportID() {
  unsigned int old = h_nextTransportID;
  h_nextTransportID++;
  return old;
}

// A function to shuffle two arrays with the same permutation, so that
// information can be kept together during the shuffle.
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

__host__ void shufflefloat(unsigned int *array1, float *array2, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned int t1 = array1[j];
      float t2 = array2[j];
      array1[j] = array1[i];
      array2[j] = array2[i];
      array1[i] = t1;
      array2[i] = t2;
    }
  }
}

// A function that returns the day of the week given an iteration number of
// increments of 5 minutes, in the form Sunday = 0, Monday = 1 etc.
__device__ unsigned int dayofweek(unsigned int step) {
  return (step % 2016) / 288;
}

// A struct to represent a time of day, and a function that returns a time of
// day given an iteration number of increments of 5 minutes.
struct Time {
  unsigned int hour;
  unsigned int minute;
};

__device__ struct Time timeofday(unsigned int step) {
  unsigned int hour = (step % 288) / 12;
  unsigned int minute = (step % 12) * 5;
  Time t = {hour, minute};
  return t;
}

__device__ float device_exp(float x) {
  float y = exp(x);
  return y;
}

// The function called at the beginning of the program on the CPU, to initialise
// all agents and their corresponding variables.
__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
  printf("Set TIME_STEP = %f\n", *get_TIME_STEP());
  printf("Set MAX_AGE = %u\n", *get_MAX_AGE());
  printf("Set STARTING_POPULATION = %u\n", (int)*get_STARTING_POPULATION());

  // Set all of the variables that are input as parameters in the input.xml
  // file, so that they are available in functions later.
  float church_beta0 = *get_CHURCH_BETA0();
  float church_beta1 = *get_CHURCH_BETA1();

  float church_prob0 = *get_CHURCH_PROB0();
  float church_prob1 = *get_CHURCH_PROB1();
  float church_prob2 = *get_CHURCH_PROB2();
  float church_prob3 = *get_CHURCH_PROB3();
  float church_prob4 = *get_CHURCH_PROB4();
  float church_prob5 = *get_CHURCH_PROB5();
  float church_prob6 = *get_CHURCH_PROB6();

  float church_p1 = *get_CHURCH_P1();
  float church_p2 = *get_CHURCH_P2();

  unsigned int church_k1 = *get_CHURCH_K1();
  unsigned int church_k2 = *get_CHURCH_K2();
  unsigned int church_k3 = *get_CHURCH_K3();

  float church_duration = *get_CHURCH_DURATION();

  float transport_beta0 = *get_TRANSPORT_BETA0();
  float transport_beta1 = *get_TRANSPORT_BETA1();

  float transport_freq0 = *get_TRANSPORT_FREQ0();
  float transport_freq2 = *get_TRANSPORT_FREQ2();

  float transport_dur20 = *get_TRANSPORT_DUR20();
  float transport_dur45 = *get_TRANSPORT_DUR45();

  unsigned int transport_size = *get_TRANSPORT_SIZE();

  float hiv_prevalence = *get_HIV_PREVALENCE();
  float art_coverage = *get_ART_COVERAGE();
  float rr_hiv = *get_RR_HIV();
  float rr_art = *get_RR_ART();
  float tb_prevalence = *get_TB_PREVALENCE();

  srand(0);

  // Initialise all of the agent types with an id of 1 and allocating an array
  // of memory for each one.
  h_nextID = 1;

  h_agent_AoS = h_allocate_agent_Person_array(h_agent_AoS_MAX);

  h_nextHouseholdID = 1;

  h_household_AoS = h_allocate_agent_Household_array(h_household_AoS_MAX);

  h_nextChurchID = 1;

  h_church_AoS = h_allocate_agent_Church_array(h_church_AoS_MAX);

  h_nextTransportID = 1;

  h_transport_AoS = h_allocate_agent_Transport_array(h_transport_AoS_MAX);

  // Initialise the input file generated by Python preprocessing, from which
  // the data about people will be read in.
  char const *const fileName = "data.in";
  FILE *file = fopen(fileName, "r");
  char line[256];

  // Read in the total number of people, household sizes and age/gender
  // categories, as these will be how many times we need to loop.
  unsigned int total = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int sizes = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int categories = strtol(fgets(line, sizeof(line), file), NULL, 0);
  unsigned int adults = strtol(fgets(line, sizeof(line), file), NULL, 0);

  float hivprob = (hiv_prevalence * *get_STARTING_POPULATION()) / adults;

  // Create variables to store all the necessary information about our people.
  unsigned int gender;
  unsigned int minage;

  char *maximumage;
  unsigned int maxage;

  unsigned int currentsize;
  float amount;
  unsigned int rounded;
  unsigned int age;

  // Create an array to store the number of people who live in households of
  // each size, and start each size at 0 people; also create an array of the
  // ages of people with given ids, used for church generation.
  unsigned int sizearray[32];
  signed int count;
  unsigned int ages[h_agent_AoS_MAX];

  unsigned int daycount = 0;
  unsigned int transport[h_agent_AoS_MAX];
  unsigned int days[h_agent_AoS_MAX];

  unsigned int tbarray[h_agent_AoS_MAX];
  float weights[h_agent_AoS_MAX];

  for (unsigned int i = 0; i < 32; i++) {
    sizearray[i] = 0;
  }

  // This loop runs once for each age/gender category, so once for every row in
  // the histogram.
  for (unsigned int i = 0; i < categories; i++) {

    // Read in the age/gender category we are working with. (Max age can be
    // infinity, in which case we set this to the max age parameter given in the
    // input.)
    gender = strtol(fgets(line, sizeof(line), file), NULL, 0);
    minage = strtol(fgets(line, sizeof(line), file), NULL, 0);

    maximumage = fgets(line, sizeof(line), file);

    if (not strcmp(maximumage, "Inf\n")) {
      maxage = (*get_MAX_AGE());
    } else {
      maxage = strtol(maximumage, NULL, 0);
    }

    // This loop runs once for each size of household, so once for every column
    // in the histogram. At this point we are working with individual values
    // from the histogram.
    for (unsigned int j = 0; j < sizes; j++) {

      // Read in the household size we are working with and the relevant value
      // from the histogram.
      currentsize = strtol(fgets(line, sizeof(line), file), NULL, 0);
      amount = strtof(fgets(line, sizeof(line), file), NULL);

      // Adjust this value proportionally so that our final population of people
      // will match the starting population specified in the input.
      rounded = round((amount / total) * *get_STARTING_POPULATION());

      // This loop runs once for each individual person, and so this is where we
      // generate the person agents.
      for (unsigned int k = 0; k < rounded; k++) {

        // Pick a random float between 0 and 1, used for deciding whether the
        // person is a transport user.
        float random = ((float)rand() / (RAND_MAX));
        float rr_as;
        float weight;

        // Allocate memory for the agent we are generating.
        xmachine_memory_Person *h_person = h_allocate_agent_Person();

        // Pick a random age for the person between the bounds of the age
        // interval they belong to.
        age = (rand() % (maxage - minage)) + minage;

        if (gender == 1) {
          if (age >= 46) {
            rr_as = 0.50;
          } else if (age >= 26) {
            rr_as = 1.25;
          } else if (age >= 18) {
            rr_as = 1.00;
          } else {
            rr_as = 0.00;
          }
        } else {
          if (age >= 46) {
            rr_as = 1.25;
          } else if (age >= 26) {
            rr_as = 3.75;
          } else if (age >= 18) {
            rr_as = 1.00;
          } else {
            rr_as = 0.00;
          }
        }

        // Assign the variables for the person agent based on information from
        // the histogram.
        h_person->id = getNextID();
        h_person->age = age;
        h_person->gender = gender;
        h_person->householdsize = currentsize;
        h_person->busy = 0;

        // Decide whether the person is a transport user based on given input
        // probabilities.
        float useprob =
            1.0 / (1 + exp(-transport_beta0 - (transport_beta1 * age)));

        if (random < useprob) {
          h_person->transportuser = 1;
        } else {
          h_person->transportuser = 0;
        }

        // If the person is a transport user, pick a transport frequency and
        // duration for them based on input probabilities; otherwise, set these
        // variables to a dummy value.
        if (h_person->transportuser) {
          random = ((float)rand() / (RAND_MAX));

          if (random < transport_freq0) {
            h_person->transportfreq = 0;
          } else if (random < transport_freq2) {
            h_person->transportfreq = 2;
            h_person->transportday1 = (rand() % 5) + 1;
            h_person->transportday2 = -1;

            transport[daycount] = h_person->id;
            days[daycount] = h_person->transportday1;

            daycount++;
          } else {
            h_person->transportfreq = 4;
            h_person->transportday1 = (rand() % 5) + 1;
            h_person->transportday2 = h_person->transportday1;

            while (h_person->transportday2 == h_person->transportday1) {
              h_person->transportday2 = (rand() % 5) + 1;
            }

            transport[daycount] = h_person->id;
            days[daycount] = h_person->transportday1;

            daycount++;

            transport[daycount] = h_person->id;
            days[daycount] = h_person->transportday2;

            daycount++;
          }
        } else {
          h_person->transportfreq = -1;
          h_person->transport = -1;
          h_person->transportday1 = -1;
          h_person->transportday2 = -1;
        }

        random = ((float)rand() / (RAND_MAX));

        if ((random < hivprob) && (h_person->age >= 18)) {
          h_person->hiv = 1;

          random = ((float)rand() / (RAND_MAX));
          if (random < art_coverage) {
            h_person->art = 1;
            weight = rr_as * rr_hiv * rr_art;
          } else {
            h_person->art = 0;
            weight = rr_as * rr_hiv;
          }
        } else {
          h_person->hiv = 0;
          h_person->art = 0;
          weight = rr_as;
        }

        weights[h_person->id] = weight;

        // Update the arrays of information with this person's household size
        // and age.
        sizearray[currentsize]++;
        ages[h_person->id] = age;

        // Generate the agent and free them from memory on the host.
        h_person->step = 0;
        h_person->householdtime = 0;
        h_person->churchtime = 0;
        h_person->transporttime = 0;
        h_add_agent_Person_default(h_person);

        h_free_agent_Person(&h_person);
      }
    }
  }

  shuffle(transport, days, h_agent_AoS_MAX);

  // Set a counter for our current position in the array of person ids, to
  // keep track as we generate households.
  count = 0;
  float churchprob;
  total = get_agent_Person_default_count();
  unsigned int order[total];

  // Populate the array of person ids with ids up to the total number of people,
  // and then shuffle it so households are assigned randomly.
  for (unsigned int i = 0; i < total; i++) {
    order[i] = i;
  }

  shuffle(order, ages, total);

  for (unsigned int i = 0; i < total; i++) {
    tbarray[i] = i;
  }

  shufflefloat(tbarray, weights, total);

  unsigned int weightsum = 0;

  for (unsigned int i = 0; i < total; i++) {
    weightsum += weights[i];
  }

  for (unsigned int i = 0; i < total; i++) {
    float randomfloat = ((float)rand() / (RAND_MAX));

    if (randomfloat < tb_prevalence) {

      float randomweight = weightsum * ((float)rand() / (RAND_MAX));

      for (unsigned int j = 0; j < total; j++) {
        if (randomweight < weights[j]) {

          xmachine_memory_TBAssignment *h_tbassignment =
              h_allocate_agent_TBAssignment();

          h_tbassignment->id = tbarray[j];
          weights[j] = 0.0;

          h_add_agent_TBAssignment_tbdefault(h_tbassignment);

          h_free_agent_TBAssignment(&h_tbassignment);
        }

        randomweight -= weights[j];
      }
    }
  }

  // Create an array to keep track of how many adults are in each household, for
  // use when generating churches.
  unsigned int adult[h_household_AoS_MAX];
  unsigned int adultcount;

  // This loop runs once for each possible size of household.
  for (unsigned int i = 1; i < 32; i++) {

    // This loop runs once for each individual household, as calculated from the
    // number of people living in households of each size.
    for (unsigned int j = 0; j < (sizearray[i] / i); j++) {

      // Allocate memory for the household agent.
      xmachine_memory_Household *h_household = h_allocate_agent_Household();
      churchprob = 1 / (1 + exp(-church_beta0 - church_beta1 * i));
      adultcount = 0;

      // Set the household's id and size.
      h_household->id = getNextHouseholdID();
      h_household->size = i;

      // Decide if the household is a churchgoing household, based on given
      // input probabilities.
      float random = ((float)rand() / (RAND_MAX));

      if (random < churchprob) {
        h_household->churchgoing = 1;
      } else {
        h_household->churchgoing = 0;
      }

      // If the household is churchgoing, decide how frequently they go based on
      // input probabilities; if not, set this variable to a dummy value.
      if (h_household->churchgoing) {
        random = ((float)rand() / (RAND_MAX));
        if (random < church_prob0) {
          h_household->churchfreq = 0;
        } else if (random < church_prob1) {
          h_household->churchfreq = 1;
        } else if (random < church_prob2) {
          h_household->churchfreq = 2;
        } else if (random < church_prob3) {
          h_household->churchfreq = 3;
        } else if (random < church_prob4) {
          h_household->churchfreq = 4;
        } else if (random < church_prob5) {
          h_household->churchfreq = 5;
        } else if (random < church_prob6) {
          h_household->churchfreq = 6;
        } else {
          h_household->churchfreq = 7;
        }
      } else {
        h_household->churchfreq = 0;
      }

      // Allocate individual people to the household until it is full, keeping
      // track of how many of them are adults.
      for (unsigned int k = 0; k < i; k++) {
        xmachine_memory_HouseholdMembership *h_hhmembership =
            h_allocate_agent_HouseholdMembership();
        h_hhmembership->household_id = h_household->id;
        h_hhmembership->person_id = order[count];
        h_hhmembership->churchgoing = h_household->churchgoing;
        h_hhmembership->churchfreq = h_household->churchfreq;
        h_hhmembership->household_size = h_household->size;
        h_household->people[k] = order[count];
        count++;

        if (ages[count] >= 15) {
          adultcount++;
        }

        h_add_agent_HouseholdMembership_hhmembershipdefault(h_hhmembership);

        h_free_agent_HouseholdMembership(&h_hhmembership);
      }

      // Set the variable for how many adults belong in the household, generate
      // the agent and then free it from memory on the host.
      h_household->adults = adultcount;
      adult[h_household->id] = adultcount;

      h_household->step = 0;
      h_add_agent_Household_hhdefault(h_household);

      h_free_agent_Household(&h_household);
    }
  }

  // Generate an array of household ids and then shuffle it, for use when
  // generating churches and other buildings.
  unsigned int hhtotal = get_agent_Household_hhdefault_count();
  unsigned int hhorder[hhtotal];

  for (unsigned int i = 0; i < hhtotal; i++) {
    hhorder[i] = i;
  }

  shuffle(hhorder, adult, hhtotal);

  // Set a variable to keep track of our current position in this array.
  unsigned int hhposition = 0;
  unsigned int capacity;

  // This loop runs until all households have been assigned to a church.
  while (hhposition < hhtotal) {

    // Allocate memory for the church agent, and set a variable to keep track of
    // how many adults have been assigned to it.
    xmachine_memory_Church *h_church = h_allocate_agent_Church();
    capacity = 0;

    h_church->id = getNextChurchID();

    // Decide what size the church is, based on given input probabilities.
    float random = ((float)rand() / (RAND_MAX));

    if (random < church_p1) {
      h_church->size = church_k1;
    } else if (random < church_p2) {
      h_church->size = church_k2;
    } else {
      h_church->size = church_k3;
    }

    // Decide whether the church services will be 1.5 hours or 3.5 hours, based
    // on the input probability.
    random = ((float)rand() / (RAND_MAX));

    if (random < church_duration) {
      h_church->duration = 1.5;
    } else {
      h_church->duration = 3.5;
    }

    // Allocate households to the church until it has reached its capacity of
    // adults, as defined by the size of the church.
    count = 0;

    while (capacity < h_church->size && hhposition < hhtotal) {
      xmachine_memory_ChurchMembership *h_chumembership =
          h_allocate_agent_ChurchMembership();
      h_chumembership->church_id = h_church->id;
      h_chumembership->household_id = hhorder[hhposition];
      h_chumembership->churchdur = h_church->duration;

      h_church->households[count] = hhorder[hhposition];
      count++;
      capacity += adult[hhposition];
      hhposition++;

      h_add_agent_ChurchMembership_chumembershipdefault(h_chumembership);

      h_free_agent_ChurchMembership(&h_chumembership);
    }

    // Generate the church agent and free it from memory on the host.
    h_church->step = 0;
    h_add_agent_Church_chudefault(h_church);

    h_free_agent_Church(&h_church);
  }

  for (unsigned int i = 1; i <= 5; i++) {

    unsigned int currentday = 0;
    unsigned int currentpeople[h_agent_AoS_MAX];

    for (unsigned int j = 0; j < h_agent_AoS_MAX; j++) {
      if (days[j] == i) {
        currentpeople[currentday] = transport[j];
        currentday++;
      }
    }

    unsigned int countdone = 0;
    unsigned int capacity;

    while (countdone < currentday) {
      xmachine_memory_Transport *h_transport = h_allocate_agent_Transport();
      capacity = 0;

      h_transport->id = getNextTransportID();
      h_transport->day = i;

      float random = ((float)rand() / (RAND_MAX));

      if (random < transport_dur20) {
        h_transport->duration = 20;
      } else if (random < transport_dur45) {
        h_transport->duration = 45;
      } else {
        h_transport->duration = 60;
      }

      while (capacity < transport_size && countdone < currentday) {

        h_transport->people[capacity] = currentpeople[countdone];

        xmachine_memory_TransportMembership *h_trmembership =
            h_allocate_agent_TransportMembership();
        h_trmembership->person_id = h_transport->people[capacity];
        h_trmembership->transport_id = h_transport->id;
        h_trmembership->duration = h_transport->duration;

        h_add_agent_TransportMembership_trmembershipdefault(h_trmembership);

        h_free_agent_TransportMembership(&h_trmembership);

        capacity++;
        countdone++;
      }

      h_transport->step = 0;
      h_add_agent_Transport_trdefault(h_transport);

      h_free_agent_Transport(&h_transport);
    }
  }

  while (fgets(line, sizeof(line), file)) {
    printf("%s", line);
  }

  printf("Sizes = %u\n", sizes);
  printf("Categories = %u\n", categories);

  fclose(file);
}

// Function that prints out the number of agents generated after initialisation.
__FLAME_GPU_INIT_FUNC__ void generateAgentsInit() {

  printf("Population after init function: %u\n",
         get_agent_Person_default_count());
}

// Function that prints out the number of agents after each iteration.
__FLAME_GPU_STEP_FUNC__ void generatePersonStep() {

  // printf("Population after step function %u\n",
  //        get_agent_Person_default_count());
}

// Function for generating output data in csv files, which runs after every
// iteration and saves data whenever specified.
__FLAME_GPU_EXIT_FUNC__ void customOutputFunction() {

  // Assign a variable for the directory where our files will be output, and
  // check which iteration we are currently on.
  const char *directory = getOutputDir();

  // If there is new information about the person agents to output, this code
  // creates a csv file and outputs data about people and their variables to
  // that file.
  std::string outputFilename =
      std::string(std::string(directory) + "person-output.csv");

  FILE *fp = fopen(outputFilename.c_str(), "w");

  if (fp != nullptr) {
    fprintf(stdout, "Outputting some Person data to %s\n",
            outputFilename.c_str());

    fprintf(
        fp,
        "ID, gender, age, household_size, hiv, art, time_home, time_church, "
        "time_transport\n");

    for (int index = 0; index < get_agent_Person_s2_count(); index++) {

      fprintf(fp, "%u, %u, %u, %u, %u, %u, %u, %u, %u\n",
              get_Person_s2_variable_id(index),
              get_Person_s2_variable_gender(index),
              get_Person_s2_variable_age(index),
              get_Person_s2_variable_householdsize(index),
              get_Person_s2_variable_hiv(index),
              get_Person_s2_variable_art(index),
              get_Person_s2_variable_householdtime(index),
              get_Person_s2_variable_churchtime(index),
              get_Person_s2_variable_transporttime(index));
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

// At the end of the run, free all of the agents from memory and output the
// final population of people.
__FLAME_GPU_EXIT_FUNC__ void exitFunction() {

  h_free_agent_Person_array(&h_agent_AoS, h_agent_AoS_MAX);

  printf("Population for exit function: %u\n", get_agent_Person_s2_count());
}

// The update functions for each agent type, which are involved in deciding
// where a person is at a given time.
__FLAME_GPU_FUNC__ int update(xmachine_memory_Person *person,
                              xmachine_message_location_list *location_messages,
                              RNG_rand48 *rand48) {

  // float random = rnd<CONTINUOUS>(rand48);

  unsigned int day = dayofweek(person->step);
  struct Time t = timeofday(person->step);
  unsigned int hour = t.hour;
  unsigned int minute = t.minute;

  if (person->busy == 0) {
    if (hour == 14 && minute == 0 && person->church != -1) {
      if (person->churchfreq == 0) {
        float random = rnd<CONTINUOUS>(rand48);
        float prob = 1 - device_exp(-6.0 / 365);

        if (random < prob) {
          person->startstep = person->step;
          person->busy = 1;
          person->location = 1;
          person->locationid = person->church;
        } else {
          person->location = 0;
          person->locationid = person->household;
        }
      } else if (person->churchfreq > day) {
        person->startstep = person->step;
        person->busy = 1;
        person->location = 1;
        person->locationid = person->church;
      }
    } else if (person->transportdur != 0 &&
               (day == person->transportday1 || day == person->transportday2)) {
      if ((hour == 7 && minute == 0) || (hour == 17 && minute == 0)) {
        person->startstep = person->step;
        person->busy = 1;
        person->location = 2;
        person->locationid = person->transport;
      } else {
        person->location = 0;
        person->locationid = person->household;
      }
    } else {
      person->location = 0;
      person->locationid = person->household;
    }
  } else {
    if (person->location == 1 &&
        (float)(person->step - person->startstep) >= person->churchdur * 12) {
      person->busy = 0;
      person->location = 0;
      person->locationid = person->household;
    } else if (person->location == 2 &&
               (float)(person->step - person->startstep) >=
                   person->transportdur / 5) {
      person->busy = 0;
      person->location = 0;
      person->locationid = person->household;
    }
  }

  person->step += TIME_STEP;

  if (person->location == 0) {
    person->householdtime += 5 * TIME_STEP;
  } else if (person->location == 1) {
    person->churchtime += 5 * TIME_STEP;
  } else if (person->location == 2) {
    person->transporttime += 5 * TIME_STEP;
  }

  add_location_message(location_messages, person->id, person->location,
                       person->locationid, day, hour, minute);

  return 0;
}

__FLAME_GPU_FUNC__ int hhupdate(xmachine_memory_Household *household) {
  household->step += TIME_STEP;
  return 0;
}

__FLAME_GPU_FUNC__ int chuupdate(xmachine_memory_Church *church) {
  church->step += TIME_STEP;
  return 0;
}

__FLAME_GPU_FUNC__ int trupdate(xmachine_memory_Transport *transport) {
  transport->step += TIME_STEP;
  return 0;
}

__FLAME_GPU_FUNC__ int
tbinit(xmachine_memory_TBAssignment *tbassignment,
       xmachine_message_tb_assignment_list *tb_assignment_messages) {
  add_tb_assignment_message(tb_assignment_messages, tbassignment->id);
  return 1;
}

__FLAME_GPU_FUNC__ int trinit(
    xmachine_memory_TransportMembership *trmembership,
    xmachine_message_transport_membership_list *transport_membership_messages) {
  add_transport_membership_message(
      transport_membership_messages, trmembership->person_id,
      trmembership->transport_id, trmembership->duration);
  return 1;
}

__FLAME_GPU_FUNC__ int
chuinit(xmachine_memory_ChurchMembership *chumembership,
        xmachine_message_church_membership_list *church_membership_messages) {
  add_church_membership_message(
      church_membership_messages, chumembership->church_id,
      chumembership->household_id, chumembership->churchdur);
  return 1;
}

__FLAME_GPU_FUNC__ int hhinit(
    xmachine_memory_HouseholdMembership *hhmembership,
    xmachine_message_church_membership_list *church_membership_messages,
    xmachine_message_household_membership_list *household_membership_messages) {

  int churchid = -1;
  float churchdur = 0;
  xmachine_message_church_membership *church_membership_message =
      get_first_church_membership_message(church_membership_messages);
  unsigned int householdid = hhmembership->household_id;

  while (church_membership_message) {
    if (church_membership_message->household_id == householdid &&
        hhmembership->churchgoing) {
      churchid = (int)church_membership_message->church_id;
      churchdur = church_membership_message->churchdur;
    }
    church_membership_message = get_next_church_membership_message(
        church_membership_message, church_membership_messages);
  }
  add_household_membership_message(
      household_membership_messages, hhmembership->household_id,
      hhmembership->person_id, hhmembership->household_size, churchid,
      hhmembership->churchfreq, churchdur);
  return 1;
}

__FLAME_GPU_FUNC__ int persontrinit(
    xmachine_memory_Person *person,
    xmachine_message_transport_membership_list *transport_membership_messages) {
  unsigned int personid = person->id;
  xmachine_message_transport_membership *transport_membership_message =
      get_first_transport_membership_message(transport_membership_messages);

  while (transport_membership_message) {
    if (transport_membership_message->person_id == personid) {
      person->transport = transport_membership_message->transport_id;
      person->transportdur = transport_membership_message->duration;
    } else {
    }
    transport_membership_message = get_next_transport_membership_message(
        transport_membership_message, transport_membership_messages);
  }

  return 0;
}
__FLAME_GPU_FUNC__ int personhhinit(
    xmachine_memory_Person *person,
    xmachine_message_household_membership_list *household_membership_messages) {
  xmachine_message_household_membership *household_membership_message =
      get_first_household_membership_message(household_membership_messages);
  unsigned int personid = person->id;

  while (household_membership_message) {
    if (household_membership_message->person_id == personid) {
      person->household = household_membership_message->household_id;
      person->householdsize = household_membership_message->household_size;
      person->church = household_membership_message->church_id;
      person->churchfreq = household_membership_message->churchfreq;
      person->churchdur = household_membership_message->churchdur;
    }
    household_membership_message = get_next_household_membership_message(
        household_membership_message, household_membership_messages);
  }
  return 0;
}

#endif
