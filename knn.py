import math, numpy as np

labels = np.array([])
train = np.array([])
test = np.array([])
optimize = [0, 0, 0]

def main():
    global train, test, optimize
    # default k value
    k = 100

    # bring in train and test data
    loadData()

    # search for best avg, variance in given list of k sizes
    sizes = [1]
    for i in range(50, 501, 50):
        sizes.append(i)
    print("Optimizing k for sizes\n", sizes)
    hyperSearch(train, sizes)
    k = optimize[2]

    # use train to infer test set with k = optimized k
    predictions = knn(train, test, k)
    savePrediction(predictions)

    # do hyperparameter search
    sizes = [1,3,5,7,9,99,999,6000]
    for size in sizes:
        out = open("hyperParamSearch.csv", "a")
        out.write("K = ")
        out.write(str(size))
        out.write("\n")

        if size == 6000:
            data = checkTrainAccuracy(train, train, 8000)
        else:
            data = checkTrainAccuracy(train, train, size)
        out.write("\tTrain Accuracy: ")
        out.write("\t")
        out.write(str(data))
        out.write("%")
        out.write("\n")

        data = knnCrossValidate(train, size)
        out.write("\t4-Fold Average: ")
        out.write("\t")
        out.write(str(data[0]))
        out.write("\n")
        out.write("\t")
        out.write("Variance: ")
        out.write("\t\t\t")
        out.write(str(data[1]))
        out.write("\n")
        out.write("---------------------------------\n")

def hyperSearch(train, ksizes):
    global optimize
    accuracies = []
    out = open("hyper_param.csv", "a")
    
    print("=============================")
    print("====Hyperparameter Search====")
    print("=============================")
    out.write("avg, var\n")
    for k in ksizes:
        data = knnCrossValidate(train, k)
        if data[0] > optimize[0]:
            optimize = data
        out.write("k = ")
        out.write(str(data[2]))
        out.write("\n")
        out.write(str(data[0]))
        out.write(",")
        out.write(str(data[1]))
        out.write("\n")
    out.write
    print("Optimized k = ", optimize[2])

def knn(train, test, k):
    output = []
    count = 0
    for i in range(len(test)):
        normsVector = calcDistance(test[i], train)
        neighbors = findNeighbors(normsVector, k)
        income = determineIncome(neighbors, train, k)
        if income == 1:
            count += 1
        output.append([int(test[i][0]), income])
    return output

def knnCrossValidate(train, k):
    accuracy = np.array([0,0,0,0])
    print("Cross Verify with k = ", k)
    for i in range(4):
        print("Fold ", i + 1)
        test = np.array([])
        crossTrain = np.array([])
        # test:0000-1999 train:2000-7999
        if i == 0:
            test = train[0:2000]
            crossTrain = train[2000:]
        # test:2000-3999 train:0-1999, 4000-7999
        elif i == 1:
            test = train[2000:4000]
            crossTrain = np.vstack((train[0:2000], train[4000:]))
        # test: 4000-5999 train:0-3999, 6000-7999
        elif i == 2:
            test = train[4000:6000]
            crossTrain = np.vstack((train[0:4000], train[6000:]))
        # test: 6000-7999 train:0-5999
        else:
            test = train[6000:8000]
            crossTrain = train[0:6000]

        output = []
        count = 0
        for j in range(len(test)):
            actualIncome = test[j][-1]
            testPoint = test[j][:-1]
            normsVector = calcDistance(testPoint, crossTrain)
            neighbors = findNeighbors(normsVector, k)
            income = determineIncome(neighbors, crossTrain, k)
            if income == actualIncome:
                count += 1
        accuracy[i] = float(count/len(test)) * 100
        print("==>\t", accuracy[i], "%")
    average = np.average(accuracy)
    variance = np.var(accuracy)
    print("\tAverage: ", average,"%\tVariance: ", variance, "\n")
    return [average, variance, k]

def checkTrainAccuracy(train, test, k):
    count = 0
    for i in range(len(test)):
        actualIncome = test[i][-1]
        testPoint = test[i][:-1]
        normsVector = calcDistance(testPoint, train)
        neighbors = findNeighbors(normsVector, k)
        income = determineIncome(neighbors, train, k)
        if income == actualIncome:
            count += 1
    accuracy = float(count / len(test)) * 100
    print("Train Accuracy with k = ", k, "\n", accuracy, "%")
    return accuracy

def loadData():
    global labels, train, test

    labels = np.loadtxt("train.csv", dtype=str, delimiter=",", max_rows=1)
    train = np.loadtxt("train.csv", dtype=float, delimiter=",", skiprows=1)
    test = np.loadtxt("test_pub.csv", dtype=float, delimiter=",", skiprows=1)

def calcDistance(datapoint, train):
    normsData = train[:, 1:-1]
    normsPoint = datapoint[1:]
    normsVector = np.linalg.norm(normsData - normsPoint, axis=1)
    return normsVector

def findNeighbors(vector, k):
    sortedIndices = np.argsort(vector)
    neighbors = []
    for i in range(k):
        neighbors.append(sortedIndices[i])
    return neighbors

def determineIncome(neighbors, train, k):
    vote = 0
    for i in range(k):
        vote += train[neighbors[i]][len(train[i]) - 1]
    if vote < math.ceil(k / 2):
        return 0
    else:
        return 1

def savePrediction(output):
    out = open("out.csv", "w")
    out.write("id,income\n")
    for point in output:
        out.write(str(point[0]))
        out.write(",")
        out.write(str(point[1]))
        out.write("\n")

main()