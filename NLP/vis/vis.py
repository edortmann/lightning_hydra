import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    # Sample data (replace with your real data)
    #margin = np.random.uniform(0.1, 0.3, 100)
    #accuracy_difference = np.abs(margin - np.random.uniform(0, 1, 100))
    #weight_decay = np.random.uniform(0.02, 0.18, 100)

    df = pd.read_csv('bert_vis_data.csv')

    margin = df['margin'].to_numpy()
    accuracy_difference = df['train-test acc'].to_numpy()
    weight_decay = df['weight_decay'].to_numpy()

    # Polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(margin.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_poly, accuracy_difference)
    y_poly_pred = poly_reg_model.predict(X_poly)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter plot with color map for weight decay
    scatter = plt.scatter(margin, accuracy_difference, c=weight_decay, cmap='viridis')

    # Plot the polynomial regression line
    plt.plot(margin, y_poly_pred, color='red', label='Polynomial Regression Line', linewidth=2)

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Weight Decay')

    # Labels and title
    plt.xlabel('Margin (Inverted Frobenius Norm)')
    plt.ylabel('Absolute Train-Test Accuracy Difference')
    plt.title('Margin vs Absolute Train-Test Accuracy Difference with Polynomial Regression')

    # Add legend
    plt.legend()

    # Show plot
    plt.savefig('vis.png')