import pandas

data = pandas.read_csv('histo.csv')

for index, row in data.iterrows():
  print(index)
  print(row)

