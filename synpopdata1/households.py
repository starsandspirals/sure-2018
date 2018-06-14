import pandas
import random

data = pandas.read_csv('histo.csv')

sexage = data['sexage']
colsums = data.sum(axis=0)[1:]

household_sizes = list (map (int, colsums.index.values))
households = colsums.divide (household_sizes)
households = households.astype(int)

rowsums = data.sum(axis=1)
people = pandas.Series (rowsums.tolist(), index=sexage)

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

random.shuffle(row_labels)

result = {}
count = 0

for x in col_labels:

  size = int(col_sizes[x])
  list = []

  for y in range (0, size):

    person = row_labels[count]
    list.append(person)
    count += 1

  result.update({x: list})

output = pandas.DataFrame(0, index=row_labels, columns=col_labels)

print(result)
print(output)