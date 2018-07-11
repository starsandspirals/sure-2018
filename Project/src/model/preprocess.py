import sys
import pandas


def generate(s):
    data = pandas.read_csv('src/model/histo.csv')
    fileout = open(s, 'w')
    categories, sizes = data.shape
    colsums = data.sum(axis=0)[1:]
    total = colsums.sum()
    adults = 0

    fileout.write(str(total) + '\n')

    fileout.write(str(sizes - 1) + '\n')
    fileout.write(str(categories) + '\n')

    for index, row in data.iterrows():
        indices, columns = [], []
        category = row['sexage'].split(':')
        gender, agegroup = category[0], category[1].split(',')
        minage, maxage = agegroup[0][1:], agegroup[1][:-1]

        for index, col in row[1:].iteritems():
            indices.append(index)
            columns.append(col)

        for n in range(0, len(indices)):
            if int(minage) >= 15:
                adults += columns[n]

    fileout.write(str(adults) + '\n')

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
