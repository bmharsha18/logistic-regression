"""

Suppose that you are the administrator of a university department and
you want to determine each applicant’s chance of admission based on their
results on two exams. You have historical data from previous applicants
that you can use as a training set for logistic regression. For each training
example, you have the applicant’s scores on two exams and the admissions
decision.

Your task is to build a classification model that estimates an applicant’s
probability of admission based the scores from those two exams.

"""
import os
import csv
import matplotlib.pyplot as plt2
from sklearn import preprocessing
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.cross_validation import train_test_split
import numpy as np

os.system("clear")
def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
print('\nLogistic Regression to predict the applicant admission based on the exam scores!\n')
data = list()
fp=open('dataset.txt','r')
reader = csv.reader(fp , delimiter=',')
for row in reader:
    data.append(row)   
m = len(data)
x1 = list()
y1 = list()
x2 = list()
y2 = list()

for i in data:
    if i[2]=='0':
        x1.append(float(i[0]))
        y1.append(float(i[1]))
    elif i[2]=='1':
        x2.append(float(i[0]))
        y2.append(float(i[1]))

scatter(x1,y1, marker='x', c='r')
scatter(x2,y2, marker='o', c='b')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()

X = list()
Y = list()

for i in range(m):
    X.append([1])

index = 0 
for i in data:
    X[index].append(float(i[0]))
    X[index].append(float(i[1]))
    Y.append(int(i[2]))
    index+=1
    
X = min_max_scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
Y_test = np.array(Y_test).reshape(len(Y_test),1)
Y_train = np.array(Y_train).reshape(len(Y_train),1)

def costFunction(X,Y,theta):
    m = len(Y)
    temp_result = np.dot(X,theta)    
    predictions = sigmoid(temp_result)
    left_part = np.dot(-1*(np.transpose(Y)),predictions)
    right_part = np.dot(np.transpose((1-Y)),np.log(1-predictions))
    J = (1/m) * (left_part - right_part)
    return J


alpha = 0.1
iterations = 2000
X = np.array(X)
Y = np.array(Y).reshape(len(Y),1)
theta = np.zeros((len(X[0]),1))

def gradient_descent(X,Y,theta,alpha,iterations):
    m = len(Y)
    J_old_values = list()
    J_old_values.append(costFunction(X,Y,theta))
    for i in range(iterations):
        temp_result = np.dot(X,theta)
        predictions = sigmoid(temp_result)
        t1 = predictions - Y
        t2 = np.dot(np.transpose(t1),X)
        t2 = (alpha/m) * t2
        theta = theta - np.transpose(t2)
        J_old_values.append(costFunction(X,Y,theta))
    return theta,J_old_values

theta_value,J_cost = gradient_descent(X_train,Y_train,theta,alpha,iterations)


plotx = list()
ploty = list()
for i in range(len(J_cost)):
    plotx.append(i)
for i in J_cost:
    ploty.append(float(i))
plt2.xlabel('No of iterations')
plt2.ylabel('Cost function "J" ')
plt2.axis([min(plotx),max(plotx),min(ploty),max(ploty)])
plt2.plot(plotx,ploty,'b')
plt2.show()

predict = list()
counter = 0
res1 = np.dot(X_test,theta_value)
res2 = sigmoid(res1)
for i in res2:
    if i>=0.5:
        predict.append(1)
    else:
        predict.append(0)

predict = np.array(predict).reshape(len(predict),1)

for i in range(len(Y_test)):
    if(Y_test[i][0]==predict[i][0]):
        counter+=1
avg = counter/len(Y_test)
accuracy = avg*100
print("Accuracy : ",accuracy)
print("\n\n\n\n\n\n\n\n")


    


    

   




        







