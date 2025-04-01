import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm  # Import LogNorm for logarithmic scaling
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    # Load your data
    df = pd.read_csv('gnn_regression_vis_data.csv')

    margin = df['margin'].to_numpy()

    # choose one of the 2 lines depending on visualizing of absolute difference or not
    accuracy_difference = df['train-test mae'].to_numpy()
    #accuracy_difference = np.abs(df['train-test acc'].to_numpy())

    weight_decay = df['weight_decay'].to_numpy()

    # Polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(margin.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_poly, accuracy_difference)
    y_poly_pred = poly_reg_model.predict(X_poly)

    # Plotting
    plt.figure(figsize=(10, 6))

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    # plt.rcParams.update({'font.size': 18})
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Scatter plot with logarithmic color map for weight decay
    scatter = plt.scatter(margin, accuracy_difference, c=weight_decay, cmap='viridis', norm=LogNorm())

    # Plot the polynomial regression line
    plt.plot(margin, y_poly_pred, color='red', label='Polynomial Regression Line', linewidth=2)

    # Add color bar with logarithmic scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Weight Decay')

    # Labels and title
    plt.xlabel('Margin (Inverted Frobenius Norm)', fontsize=BIGGER_SIZE)
    plt.ylabel('Train-Test MAE Difference', fontsize=BIGGER_SIZE)
    #plt.ylabel('Absolute Train-Test Accuracy Difference', fontsize=BIGGER_SIZE)
    #plt.title('Margin vs Train-Test Accuracy Difference with Polynomial Regression')

    # Add legend
    plt.legend()

    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig('gnn_regression_vis.png')
