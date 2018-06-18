import pandas
import random

def generate(s):

  data = pandas.read_csv('histo.csv')

  sexage = data['sexage']
  colsums = data.sum(axis=0)[1:]

  household_sizes = list(map (int, colsums.index.values))
  households = colsums.divide(household_sizes)
  households = households.astype(int)

  rowsums = data.sum(axis=1)
  people = pandas.Series(rowsums.tolist(), index=sexage)

  col_labels = []
  col_sizes = {}
  row_labels = []

  for i, v in households.items():
    
    for x in range (1, v+1):

      label = str(x) + ", H" + str(i)
      col_labels.append(label)
      col_sizes.update({label: i})

  for i, v in people.items():

    for x in range (1, v+1):

      label = str(x) + ", " + i
      row_labels.append(label)

  row_shuffled = list(row_labels)
  random.shuffle(row_shuffled)

  assignment = {}
  count = 0

  for x in col_labels:

    size = int(col_sizes[x])
    group = []

    for y in range (0, size):

      person = row_shuffled[count]
      group.append(person)
      count += 1

    assignment.update({x: group})

  result = {}
  result.update({"sexage": row_labels})

  for x in col_labels:

    assigned = assignment[x]
    group = []

    for y in row_labels:

      if y in assigned:
        group.append(1)
      else:
        group.append(0)
    
    result.update({x: group})

  output = pandas.DataFrame.from_dict(result)

  return output

def test(x, s):

  row_labels = x.iloc[:, 0]
  col_labels = pandas.Series(x.columns.values[1:])

  print(row_labels)
  print(col_labels)



print(test(generate('histo.csv'), 'test'))