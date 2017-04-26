def convert_columns(filename):

    import csv

    filenameout = filename[:-4] + "_out" + filename[-4:]

    data = []

    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            # print(', '.join(row))
            data.append(row)

    for i in range(1, len(data[:])):
        for j in range(len(data[0][:])):
            data[i][j] = float("{0:.5f}".format(float(data[i][j])))
            if j != 0:
                data[i][j] = float("{0:.5f}".format(1. - data[i][j]))


    with open(filenameout, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for n in range(len(data)):
            spamwriter.writerow(data[n])
    # print(data)
    # print(filenameout)

# convert_columns('/home/mgessner/vm_share/Beispieldatei.txt')
import sys

if len(sys.argv) != 2:
    print('Bitte zu konvertierende Textdatei angeben und erneut versuchen!')
elif len(sys.argv) == 2:
    convert_columns(sys.argv[1])

# convert_columns('Beispieldatei.txt')