with open('E:/data/abide/s-ag11.csv') as filein, open(
        'E:/data/abide/s-ag11.txt', 'w') as fileout:
    for line in filein:
        line = line.replace(",", ":")