import numpy as np
from sklearn import svm
import libsvm
from libsvm import svm
from libsvm import svmutil
import geatpy as ea
# from libsvm.python.svm import *
# # from libsvm.python.svmutil import *
from concurrent.futures import ThreadPoolExecutor
import threading
import linecache
import time

# class MyProblem(ea.Problem):

def dealWithFile(file, startLineNo, splitNum):
    lineNo = startLineNo
    arr = np.empty((0, 7))
    while True:
        newline = linecache.getline(file, lineNo)
        lineNo += 1
        if (not newline) or (lineNo > startLineNo + splitNum):
            break
        newlist = newline.split(",")
        if newlist[6].startswith('draw'):
            newlist[6] = 1
        else:
            newlist[6] = -1
        for col in range(6):
            if col % 2 == 0:
                newlist[col] = ord(newlist[col]) - 96
            else:
                newlist[col] = ord(newlist[col]) - 48
        arr = np.append(arr, np.array(newlist).reshape(1, 7), axis=0)
    return arr

if __name__ == '__main__':
    timeStart = time.time()
    file = "D:\\Program\\KrK_0513\\krkopt.data"
    for sumLineNum, line in enumerate(open(file, 'r')):
        pass
    sumLineNum += 1
    # f = open(file, 'r')
    arr = np.empty((0, 7))
    # lineNo = 1
    splitNum = 1000
    groupNum = int(sumLineNum / splitNum) + 1
    startLineNo = 1
    for i in range(groupNum):
        arrTemp = dealWithFile(file, startLineNo, splitNum)
        startLineNo = startLineNo + splitNum
        arr = np.append(arr, arrTemp, axis=0)
    np.random.shuffle(arr)
    xapp = arr[:, 0:6]
    yapp = arr[:, 6]

    numForTrain = 5000
    numForTest = sumLineNum - numForTrain

    xForTrain = xapp[0:numForTrain, :]
    yForTrain = yapp[0:numForTrain]
    avgX = np.mean(xForTrain, axis=0)
    varX = np.var(xForTrain, axis=0)
    xForTest = xapp[numForTrain:numForTest, :]
    yForTest = yapp[numForTrain:numForTest]

    xForTrain = (xForTrain - avgX) / varX
    xForTest = (xForTest - avgX) / varX

    best_acc = 0

    for c in range(20):
        for gamma in range(18):
            cReal = pow(2, (c - 5))
            gammaReal = pow(2, (gamma - 15))

            opt = '-s 0 -t 2 -c ' + str(cReal) + ' -g ' + str(gammaReal)
            model = libsvm.svmutil.svm_train(yForTrain, xForTrain, opt)
            p_label, p_acc, p_val = libsvm.svmutil.svm_predict(yForTest, xForTest, model)
            if p_acc[0] > best_acc:
                best_acc = p_acc[0]

    timeEnd = time.time()
    print(timeEnd - timeStart)
    # print("Hello world")
    print("the best acc is " + str(best_acc))


