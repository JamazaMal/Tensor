import csv

with open(R"c:\temp\nyc\train.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readCSV:
        if i > 0:
            print([round(i) for i in map(float, row[3:7:])])
            if i > 1000:
                break
        i = i + 1

    print('Done')
