import csv

import numpy
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

weekNum = []
historicalAttendance = []
predictionRange = 1

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            weekNum.append(int(row[0]))
            historicalAttendance.append(int(row[1]))

def predict_price(weekNum, historicalAttendance, x):
    weekNum = np.reshape(weekNum, (len(weekNum), 1))  # converting to matrix of n X 1

    svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.005)  # defining the support vector regression models
    svr_rbf.fit(weekNum, historicalAttendance)  # fitting the data points in the models

    plt.scatter(weekNum, historicalAttendance, color='black', label='Data')  # plotting the initial datapoints
    plt.plot(weekNum, svr_rbf.predict(weekNum), color='red', label='RBF model')  # plotting the line made by the RBF kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(np.array(x).reshape(-1, 1))[0]

get_data('AttendanceData.csv')

predicted_price = round((predict_price(weekNum, historicalAttendance, (len(historicalAttendance) + predictionRange)) + 6), 0)
stdev = round(((numpy.std(historicalAttendance)) / 4), 1)

lowPrediction = predicted_price - stdev
highPrediction = predicted_price + stdev

print("\n")
print("Based only on historical data, the predicted attendance for the next Food Bank is " + str(predicted_price) + ".")
print("\nThe exact prediction may not be correct; however, it is extremely likely the prediction \nwill fall between " + str(lowPrediction) + " and " + str(highPrediction) + ".")
print("\nRemember to log the actual attendance number in the excel spreadsheet called 'AttendanceData.csv'.")
print("\n")
