import pandas

data = pandas.read_csv('histo.csv')
fileout = open('data.in', 'w')
sizes = len(data.columns) - 1

fileout.write(str(sizes) + '\n')

for index, row in data.iterrows():
  indices, columns = [], []
  category = row['sexage'].split(':')
  gender, agegroup = category[0], category[1].split(',')
  minage, maxage = agegroup[0][1:], agegroup[1][:-1]
  fileout.write(gender + '\n' + minage + '\n' + maxage + '\n')
  
  for index, col in row[1:].iteritems():
    indices.append(index)
    columns.append(col)

  for n in range(1, len(indices)):
    fileout.write(indices[n-1] + '\n')
    fileout.write(str(columns[n-1]) + '\n')
  
fileout.close()
  

