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

for i, v in households.items():
  
  for y in range (1, v):

    label = str(y) + ", H" + str(i)
    col_labels.append(label)

print(col_labels)
print(households)