import pandas

data = pandas.read_csv('histo.csv')
fileout = open('data.in', 'w')

for index, row in data.iterrows():
  count = 0
  indices, columns = [], []
  category = row['sexage'].split(':')
  gender, agegroup = category[0], category[1].split(',')
  minage, maxage = agegroup[0][1:], agegroup[1][:-1]
  fileout.write(gender + '\n' + minage + '\n' + maxage + '\n')
  
  for index, col in row[1:].iteritems():
    indices.append(index)
    columns.append(col)
    count += 1

  fileout.write(str(count) + '\n')

  for n in range(1, len(indices)):
    fileout.write(indices[n-1] + '\n')
    fileout.write(str(columns[n-1]) + '\n')
  


  
  

