import sys
import pandas

def generate(s):
  data = pandas.read_csv('histo.csv')
  fileout = open(s, 'w')
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
    
    for n in range(0, len(indices)):
      fileout.write(indices[n] + '\n')
      fileout.write(str(columns[n]) + '\n')
    
  fileout.close()
  
generate(sys.argv[1])

