import numpy as np
import pandas as pd
from computing import cost,gradient
from algorithm import gradient_descent
import matplotlib.pyplot as plt


def main():
    print('Gradient Descent Algorithm for Salary Prediction based on Years of Excperience')
    data = pd.read_csv('Salary_Data.csv')
    x_train = data['YearsExperience'].values
    y_train = data['Salary'].values
    
    w_initial = 0
    b_initial = 0
    iterations = 1000
    alpha = 0.05

    w_final, b_final, J_hist, p_hist = gradient_descent(x_train,y_train,w_initial,b_initial,alpha,iterations,cost,gradient)
    print(f"\n(Final w, Final b) found by gradient descent: ({w_final:8.4f},{b_final:4.4f})")

    print('Predictions : ')
    print(f"For 10 years of experience the predicted salary is: {w_final*10 + b_final:0.0f} dollars")



if __name__ == '__main__':
    main()