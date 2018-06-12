import pandas

data = pandas.read_csv('histo.csv')

sexage = data['sexage']
colsums = data.sum(axis=0)[1:]

household_sizes = list (map (int, colsums.index.values))
households = colsums.divide (household_sizes)
households = households.astype(int)

rowsums = data.sum(axis=1)
people = pandas.Series (rowsums.tolist(), index=sexage)

col_labels = []
row_labels = []

for i, v in households.items():
  
  for x in range (1, v+1):

    label = str(x) + ", H" + str(i)
    col_labels.append(label)

for i, v in people.items():

  for x in range (1, v+1):

    label = str(x) + ", " + i
    row_labels.append(label)

output = pandas.DataFrame(0, index=row_labels, columns=col_labels)

print(output)