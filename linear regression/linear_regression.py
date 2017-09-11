# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

'''
Created on 2017年9月10日
多维向量的线性回归
href:http://blog.csdn.net/LULEI1217/article/details/49386295
http://blog.csdn.net/LULEI1217/article/details/49386295
http://aeolus1983.iteye.com/blog/2344052
'''

'''
# 本实例旨在根据广告投放的不同特征实现对应销量的预测
#特征：

#TV：对于一个给定市场中单一产品，用于电视上的广告费用（以千为单位）
#Radio：在广播媒体上投资的广告费用
#Newspaper：用于报纸媒体的广告费用
#响应：

#Sales：对应产品的销量
'''

'''
# 测试pandas读取csv文件结果
# data = pd.read_csv('./Advertising.csv')
# 输出数据前五行，或者后五行，前者结果如下
# print(data.head())
# print(data.tail())
#   Unnamed: 0     TV  Radio  Newspaper  Sales
# 0           1  230.1   37.8       69.2   22.1
# 1           2   44.5   39.3       45.1   10.4
# 2           3   17.2   45.9       69.3    9.3
# 3           4  151.5   41.3       58.5   18.5
# 4           5  180.8   10.8       58.4   12.9

# pandas的两个主要数据结构：Series和DataFrame：

# Series类似于一维数组，它有一组数据以及一组与之相关的数据标签(即索引)组成。
# DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型。
# DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典。
'''
def load_data(file_name,feature_cols):
    #由于sklearn要求的是 输入X是一个特征矩阵，输出Y是一个numpy数组
    #因此X可以是pandas的DataFrame数据结构，Y可以是pandas的Series数据结构
    data = pd.read_csv(file_name)
    X = data[feature_cols]
    Y = data['Sales']
    return  X,Y

def linear_model_main():
    pass


def plot_main(Y_label, Y_predict):
    plt.figure()
    # 画图函数第一个参数表示坐标轴范围
    plt.plot(range(len(Y_label)),Y_label,'b',label = "RealValue")
    plt.plot(range(len(Y_predict)),Y_predict,'r',label = "PredictValue")
    plt.legend(loc = "upper left")
    plt.xlabel("the sequence number of sales")
    plt.ylabel('value of sales')
    plt.show()




# 计算均方误差
def get_rmse(Y_label,Y_predict):
    sum_mean = 0
    for i in range(len(Y_label)):
        sum_mean = sum_mean + (Y_label.values[i] - Y_predict[i])**2
    sum_error = np.sqrt(sum_mean / len(Y_label))
    return sum_error


if __name__ == '__main__':
    '''
    #测试dataframe数据结构的输出
    file_name = './Advertising.csv'
    feature_cols = ['TV','Radio','Newspaper']
    X,Y = load_data(file_name,feature_cols)
    print(X.shape)
    print(X.values[1:4])
    #(200, 3)
    #[[44.5   39.3   45.1]
    # [17.2   45.9   69.3]
    #[151.5   41.3   58.5]]
    '''
    file_name = './Advertising.csv'
    feature_cols = ['TV', 'Radio', 'Newspaper']
    X, Y = load_data(file_name, feature_cols)
    print(X.shape)
    print(X.values[1:4])
    model = linear_model.LinearRegression()
    a = model.fit(X.head(195),Y.head(195))
    X_test = X.tail(5)
    Y_real = Y.tail(5)
    Y_predict = model.predict(X_test)
    plot_main(Y_real,Y_predict)
    print(get_rmse(Y_real, Y_predict))



