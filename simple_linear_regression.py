import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.externals import joblib

def simple():

    data = pd.read_csv('../data/Salary_Data.csv')

    X = data.iloc[:, : -1].values
    Y = data.iloc[:, 1].values


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1/3, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)


    filename = 'finalized_model.sav'
    pickle.dump(regressor, open(filename, 'wb'))

    # Create model binary using joblib
    #joblib.dump(regressor, filename)

    loaded_model = pickle.load(open(filename, 'rb'))
    score = loaded_model.score(X_test, Y_test)

    print(score)

    list = [4]
    predictedvalue = regressor.predict(np.array(list).reshape(1, -1))
    print(predictedvalue)


def main():
    print("Hello")
    simple()

if __name__ == '__main__':
    main()