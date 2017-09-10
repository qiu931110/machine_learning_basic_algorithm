#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
import matplotlib.pyplot as plt

#np.dataset = []

#np.dataset = [( random.randint(0,500), random.randint(0,100)) for i in range(0,10,1)]
#print(np.dataset[0][0])
#a = [a for a in range(0,10,1)]
#print(a)
#x_train = [np.dataset[i][0] for i in range(0,10,1)]
#y_train = [np.dataset[i][1] for i in range(0,10,1)]
#print(x_train)
#print(y_train)
def runplt():
    plt.figure()
    plt.title('linear_regresion' )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend('point')
    #plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt
plt = runplt()
x_train = [[2],[4],[6],[8],[10],[12],[14],[16],[18],[20]]
y_train = [[2],[4],[7],[8],[10],[12],[14],[16],[19],[20]]
x_test = [[22],[25],[27],[31]]
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
#x_train= input_variables_values_training_datasets
#y_train= target_variables_values_training_datasets
#x_test= input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept
# 输出的就是模型参数，斜率（系数），和截距
print('Coefficient: n', linear.coef_)
print('Intercept: n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
print(predicted)

# plot fig
plt.plot(x_train,y_train,'b.')
plt.plot(x_test,predicted,'r-')
plt.show()

