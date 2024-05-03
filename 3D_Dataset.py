# Generating 3D make-moons data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

def Plotting_and_Saving(X, labels, save_path):
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Make Moons')
    plt.show()

    # Convert to DataFrame
    data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'Z': X[:, 2], 'Label': labels})

    # Save to Excel file
    data.to_excel(save_path, index=False)

if __name__ == '__main__':
    # Generate data_path
    save_Train_path = r".\Dataset\TrainingData.xlsx"
    save_Test_path = r".\Dataset\TestData.xlsx"

    # Generate the data (1000 datapoints)
    X_train, labels_train = make_moons_3d(n_samples=500, noise=0.2)
    X_test, labels_test = make_moons_3d(n_samples=250, noise=0.2)

    Plotting_and_Saving(X_train, labels_train, save_Train_path)
    Plotting_and_Saving(X_test, labels_test, save_Test_path)
