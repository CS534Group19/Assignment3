test = [[[3,4,5],[6,7,8],[9,10,11]]]

for i in range(len(test)):
    for j in range(len(test[i])):
        for k in range(len(test[i][j])):
            test[i][j][k] = test[i][j][k] / 2

print(test)