import pandas

data = pandas.read_csv('histo.csv')

sexage = data['sexage']
colsums = data.sum(axis=0)[1:]

household_sizes = list(map (int, colsums.index.values))
households = colsums.divide(household_sizes)
households = pandas.Series(households.tolist(), dtype=int)

rowsums = data.sum(axis=1)
people = pandas.Series(rowsums.tolist(), index=sexage)

print(households)
print(people)