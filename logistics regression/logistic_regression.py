from numpy import *
import matplotlib.pyplot as plt
import time

def sigmoid(InX):
    return 1.0 / (1 + exp(-InX))


def load_data(file_name):
    train_x = []
    train_y = []
    file = open(file_name)
    for line in file.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return  mat(train_x), mat(train_y).transpose()


def train_model(train_x, train_y, opts):

    startTime = time.time()

    numSamples, numFeature = shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones((numFeature,1))

    for k in range(maxIter):
        if opts['optimizeType'] == "gradDescent":
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
            pass
        elif opts['optimizeType'] == "stocGradDescent":
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
            pass
        elif opts['optimizeType'] == "smoothStocGradDescent":
            dataIndex = list(range(numSamples))
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex])
            pass
        else:
            raise NameError('Not support optimize method type!')


    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    return weights


def test_model(test_x,test_y,weights):
    numSamples,numFeature = shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


def show_result(test_x,test_y,weights):
    numSamples, numFeatures = shape(train_x)

    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    '''
    file_name = './testSet.txt'
    train_x ,train_y = load_data(file_name)
    print(train_x)
    print(mat(train_x))
    print(mat(train_y))
    print(mat(train_y).transpose())
    输出结果：
    [[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862],。。。。
    [[  1.00000000e+00  -1.76120000e-02   1.40530640e+01]
     [  1.00000000e+00  -1.39563400e+00   4.66254100e+00]
     [  1.00000000e+00  -7.52157000e-01   6.53862000e+00]
     [  1.00000000e+00  -1.32237100e+00   7.15285300e+00]
     [  1.00000000e+00   4.23363000e-01   1.10546770e+01]
     [  1.00000000e+00   4.06704000e-01   7.06733500e+00]
     [  1.00000000e+00   6.67394000e-01   1.27414520e+01]。。。。
     [[ 0.  1.  0.  0.  0.  1.  0.  1.  0.  0.  1.  0.  1.  0.  1.  1.  1.  1.
   1.  1.  1.  1.  0.  1.  1.  0.  0.  1.  1.  0.  1.  1.  0.  1.  1.  0.
   0.  0.  0.  0.  1.  1.  0.  1.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.
   1.  1.  0.  1.  0.  1.  1.  1.  0.  0.  0.  1.  1.  0.  0.  0.  0.  1.
   0.  1.  0.  0.  1.  1.  1.  1.  0.  1.  0.  1.  1.  1.  1.  0.  1.  1.
   1.  0.  0.  1.  1.  1.  0.  1.  0.  0.]]
   [[ 0.]
 [ 1.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 1.]。。。。

    '''
    file_name = './testSet.txt'
    train_x, train_y = load_data(file_name)
    test_x = train_x;
    test_y = train_y

    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights = train_model(train_x, train_y, opts)

    accuracy = test_model(test_x, test_y, optimalWeights)

    print('The classify accuracy is: %.3f%%' % (accuracy * 100))
    show_result(train_x, train_y, optimalWeights)