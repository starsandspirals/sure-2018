import pandas
import random

def generate(s):

  data = pandas.read_csv('histo.csv')

  sexage = data['sexage']
  colsums = data.sum(axis=0)[1:]

  household_sizes = list(map (int, colsums.index.values))
  households = colsums.divide(household_sizes)
  households = households.astype(int)
  col_dict = households.to_dict()

  rowsums = data.sum(axis=1)
  people = pandas.Series(rowsums.tolist(), index=sexage)
  row_dict = people.to_dict()

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

  return output, col_dict, row_dict

def test(x, c, r):

  row_labels = x.iloc[:, 0].tolist()
  col_labels = x.columns.values[1:].tolist()

  split_rows = map (lambda x: x.split(", "), row_labels)
  split_columns = map (lambda x: x.split(", "), col_labels)

  tuple_rows = list(map (tuple, split_rows))
  tuple_columns = list(map (tuple, split_columns))

  row_max = {}
  col_max = {}

  for (x, y) in tuple_rows:

    current_max = int(row_max.get(y, 0))

    if int(x) >= current_max:

      row_max.update({y: str(x))})

  for (x, y) in tuple_columns:

    current_max = int(col_max.get(y, 0))

    if int(x) >= current_max:

      col_max.update({y[1:]: str(x))})

  if (c == col_max and r == row_max):
    return True
  else:
    return False

output, columns, rows = generate('histo.csv')
print(output)

if test(output, columns, rows):
  print("Test succeeded!")
else:
  print("Test failed.")

