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

// A function that returns the day of the week given an iteration number of
// increments of 5 minutes, in the form Sunday = 0, Monday = 1 etc.
__host__ unsigned int dayofweek(unsigned int step) { return (step % 288) % 7; }

// A struct to represent a time of day, and a function that returns a time of
// day given an iteration number of increments of 5 minutes.
struct Time {
  unsigned int hour;
  unsigned int minute;
};

__host__ struct Time timeofday(unsigned int step) {
  unsigned int hour = (step % 12) % 24;
  unsigned int minute = (step % 12) * 5;
  Time t = {hour, minute};
  return t;
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

        // Allocate memory for the agent we are generating.
        xmachine_memory_Person *h_person = h_allocate_agent_Person();

        // Pick a random age for the person between the bounds of the age
        // interval they belong to.
        age = (rand() % (maxage - minage)) + minage;

        // Assign the variables for the person agent based on information from
        // the histogram.
        h_person->id = getNextID();
        h_person->age = age;
        h_person->gender = gender;
        h_person->householdsize = currentsize;

        // Decide whether the person is a transport user based on given input
        // probabilities.
        float useprob = 1 + exp(-transport_beta0 - transport_beta1 * age);

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
          } else {
            h_person->transportfreq = 4;
          }

          random = ((float)rand() / (RAND_MAX));

          if (random < transport_dur20) {
            h_person->transportdur = 20;
          } else if (random < transport_dur45) {
            h_person->transportdur = 45;
          } else {
            h_person->transportdur = 60;
          }
        } else {
          h_person->transportfreq = -1;
          h_person->transportdur = -1;
        }

        // Update the arrays of information with this person's household size
        // and age.
        sizearray[currentsize]++;
        ages[h_person->id] = age;

        // Generate the agent and free them from memory on the host.
        h_person->step = 0;
        h_add_agent_Person_default(h_person);

        h_free_agent_Person(&h_person);
      }
    }
  }

  // Set a counter for our current position in the array of person ids, to keep
  // track as we generate households.
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

      // Allocate individual people to the household until it is full, keeping
      // track of how many of them are adults.
      for (unsigned int k = 0; k < i; k++) {
        xmachine_memory_HouseholdMembership *h_hhmembership =
            h_allocate_agent_HouseholdMembership();
        h_hhmembership->household_id = h_household->id;
        h_hhmembership->person_id = order[count];
        h_household->people[k] = order[count];
        count++;

        if (ages[count] >= 15) {
          adultcount++;
        }

        h_add_agent_HouseholdMembership_hhmembershipdefault(h_hhmembership);

        h_free_agent_HouseholdMembership(&h_hhmembership);
      }

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

      // Set the variable for how many adults belong in the household, generate
      // the agent and then free it from memory on the host.
      h_household->adults = adultcount;
      adult[h_household->id] = adultcount;

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

      h_church->households[count] = hhorder[hhposition];
      count++;
      capacity += adult[hhposition];
      hhposition++;

      h_add_agent_ChurchMembership_chumembershipdefault(h_chumembership);

      h_free_agent_ChurchMembership(&h_chumembership);
    }

    // Generate the church agent and free it from memory on the host.
    h_add_agent_Church_chudefault(h_church);

    h_free_agent_Church(&h_church);
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
__FLAME_GPU_STEP_FUNC__ void customOutputStepFunction() {

  // Assign a variable for the directory where our files will be output, and
  // check which iteration we are currently on.
  const char *directory = getOutputDir();
  unsigned int iteration = getIterationNumber();

  // If there is new information about the person agents to output, this code
  // creates a csv file and outputs data about people and their variables to
  // that file.
  if (iteration == 1) {

    std::string outputFilename =
        std::string(std::string(directory) + "person-output-" +
                    std::to_string(iteration) + ".csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Person data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, gender, age, household, church\n");

      for (int index = 0; index < get_agent_Person_s2_count(); index++) {

        fprintf(fp, "%u, %u, %u, %u, %u\n", get_Person_s2_variable_id(index),
                get_Person_s2_variable_gender(index),
                get_Person_s2_variable_age(index),
                get_Person_s2_variable_household(index),
                get_Person_s2_variable_church(index));
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

  // At the beginning of the run, output information about households and their
  // variables to a separate csv file.
  if (iteration == 1) {

    std::string outputFilename =
        std::string(std::string(directory) + "household-output.csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Household data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, size, churchgoing, churchfreq, adults\n");

      for (int index = 0; index < get_agent_Household_hhdefault_count();
           index++) {

        fprintf(fp, "%u, %u, %u, %u, %u\n",
                get_Household_hhdefault_variable_id(index),
                get_Household_hhdefault_variable_size(index),
                get_Household_hhdefault_variable_churchgoing(index),
                get_Household_hhdefault_variable_churchfreq(index),
                get_Household_hhdefault_variable_adults(index));
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

  // Similarly, at the beginning of a run output information about churches and
  // other types of building to their own individual csv files.
  if (iteration == 1) {

    std::string outputFilename =
        std::string(std::string(directory) + "church-output.csv");

    FILE *fp = fopen(outputFilename.c_str(), "w");

    if (fp != nullptr) {
      fprintf(stdout, "Outputting some Church data to %s\n",
              outputFilename.c_str());

      fprintf(fp, "ID, size, duration\n");

      for (int index = 0; index < get_agent_Church_chudefault_count();
           index++) {
        fprintf(fp, "%u, %u, %f\n", get_Church_chudefault_variable_id(index),
                get_Church_chudefault_variable_size(index),
                get_Church_chudefault_variable_duration(index));
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

// At the end of the run, free all of the agents from memory and output the
// final population of people.
__FLAME_GPU_EXIT_FUNC__ void exitFunction() {

  h_free_agent_Person_array(&h_agent_AoS, h_agent_AoS_MAX);

  printf("Population for exit function: %u\n", get_agent_Person_s2_count());
}

// The update functions for each agent type, which are involved in deciding
// where a person is at a given time.
__FLAME_GPU_FUNC__ int update(xmachine_memory_Person *person,
                              RNG_rand48 *rand48) {

  // agent->age += TIME_STEP;
  // float random = rnd<CONTINUOUS>(rand48);

  // if (random < (agent->age * SCALE_FACTOR * TIME_STEP)) {
  //   return 1;
  // }

  person->step++;

  return 0;
}

__FLAME_GPU_FUNC__ int hhupdate(xmachine_memory_Household *household) {
  return 0;
}

__FLAME_GPU_FUNC__ int chuupdate(xmachine_memory_Church *church) { return 0; }

__FLAME_GPU_FUNC__ int trupdate(xmachine_memory_Transport *transport) {
  return 0;
}

__FLAME_GPU_FUNC__ int
chuinit(xmachine_memory_ChurchMembership *chumembership,
        xmachine_message_church_membership_list *church_membership_messages) {
  add_church_membership_message(church_membership_messages,
                                chumembership->church_id,
                                chumembership->household_id);
  return 1;
}

__FLAME_GPU_FUNC__ int hhinit(
    xmachine_memory_HouseholdMembership *hhmembership,
    xmachine_message_church_membership_list *church_membership_messages,
    xmachine_message_household_membership_list *household_membership_messages) {

  xmachine_message_church_membership *church_membership_message =
      get_first_church_membership_message(church_membership_messages);
  unsigned int householdid = hhmembership->household_id;
  unsigned int churchid = 0;

  while (church_membership_message) {
    if (church_membership_message->household_id == householdid) {
      churchid = church_membership_message->church_id;
    }
    church_membership_message = get_next_church_membership_message(
        church_membership_message, church_membership_messages);
  }

  add_household_membership_message(household_membership_messages,
                                   hhmembership->household_id,
                                   hhmembership->person_id, churchid);
  return 1;
}

__FLAME_GPU_FUNC__ int init(
    xmachine_memory_Person *person,
    xmachine_message_household_membership_list *household_membership_messages) {
  xmachine_message_household_membership *household_membership_message =
      get_first_household_membership_message(household_membership_messages);
  unsigned int personid = person->id;

  while (household_membership_message) {
    if (household_membership_message->person_id == personid) {
      person->household = household_membership_message->household_id;
      person->church = household_membership_message->church_id;
    }
    household_membership_message = get_next_household_membership_message(
        household_membership_message, household_membership_messages);
  }

  return 0;
}

#endif
