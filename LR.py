# Import necessary libraries
import numpy
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer

# Function to normalize training and test data using L2 normalization
def normalize_L2(X_train, X_test):
    # Initialize a normalizer object with L2 norm
    norm = Normalizer(norm='l2')
    # Normalize the training data
    X_training_l2 = norm.transform(X_train)
    # Normalize the test data
    X_test_l2 = norm.transform(X_test)
    # Return the normalized training and test data
    return X_training_l2, X_test_l2

# Function to perform PCA (Principal Component Analysis) on training and test data
def PCA_decomposition(X_train, X_test):
    # Import the PCA module from sklearn
    from sklearn.decomposition import PCA
    # Initialize a PCA object
    pca = PCA()
    # Fit the PCA model to the training data
    pca.fit(X_train)
    # Transform the training data to its principal components
    X_train_pca = pca.transform(X_train)
    # Transform the test data to its principal components
    X_test_pca = pca.transform(X_test)
    # Return the transformed training and test data, and the PCA model
    return X_train_pca, X_test_pca, pca

# Function to fit a Logistic Regression model to the training data
def fit_LR(X_train, y_train):
    # Import the Logistic Regression module from sklearn
    from sklearn.linear_model import LogisticRegression
    # Initialize a Logistic Regression model
    lr = LogisticRegression()
    # Fit the model to the training data
    lr.fit(X_train, y_train.ravel())
    # Return the trained model
    return lr

# Function to fit a Bernoulli Naive Bayes model to the training data
def fit_Bernoulli_NB(X_train, y_train):
    # Import the BernoulliNB module from sklearn
    from sklearn.naive_bayes import BernoulliNB
    # Initialize a Bernoulli Naive Bayes model
    nb = BernoulliNB()
    # Fit the model to the training data
    nb.fit(X_train, y_train.ravel())
    # Return the trained model
    return nb


# Function to fit a K-Nearest Neighbors (KNN) model to the training data
def fit_KNN(X_train, y_train, _algorithm="", _weights="uniform"):
    # Import the neighbors module from sklearn
    from sklearn import neighbors
    # If no algorithm is specified, use the default KNN classifier
    if(_algorithm == ""):
        clf = neighbors.KNeighborsClassifier(11, weights=_weights)
    # If an algorithm is specified, use it in the KNN classifier
    else:
        clf = neighbors.KNeighborsClassifier(11, algorithm=_algorithm, weights=_weights)
    # Fit the model to the training data
    clf.fit(X_train, y_train.ravel())
    # Return the trained model
    return clf

# Function to fit a Support Vector Machine (SVM) model to the training data
def fit_SVM(X_train, y_train, _gamma="auto"):
    # Import the svm module from sklearn
    from sklearn import svm
    # Initialize an SVM classifier with the specified gamma
    clf = svm.NuSVC(gamma=_gamma)
    # Fit the model to the training data
    clf.fit(X_train, y_train.ravel())
    # Return the trained model
    return clf

# Function to plot the training data and the decision boundary of the Logistic Regression model in 2D
def plot_2D(lr, X_train, y_train):
    # Extract the first and second features
    x_f = X_train[:, 0]
    y_f = X_train[:, 1]
    # Create a new figure
    plt.figure()
    # Plot the data points of class 0 in red
    plt.plot(x_f[y_train == 0], y_f[y_train == 0], "or")
    # Plot the data points of class 1 in green
    plt.plot(x_f[y_train == 1], y_f[y_train == 1], "og")
    # Get the coefficients of the decision boundary (the line)
    thetaN = lr.coef_
    theta0 = lr.intercept_
    theta1 = thetaN[0][0]
    theta2 = thetaN[0][1]
    # Define the x range of the line
    x = numpy.array([-0.9, 0.9])
    y = -((theta0+theta1)*x)/(theta2)
    plt.plot(x, y)
    plt.show()

# Function to plot the training data and the decision boundary of the Logistic Regression model in 3D
def plot_3D(lr, X_train, y_train):
    # Import the necessary module for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    # Create a new figure
    plt.figure()
    # Add a 3D subplot
    plt.subplot(111, projection="3d")
    # Extract the first, second and third features
    x_f = X_train[:, 0]
    y_f = X_train[:, 1]
    z_f = X_train[:, 2]
    # Plot the data points of class 0 in red
    plt.plot(x_f[y_train == 0], x_f[y_train == 0], z_f[y_train == 0], "or")
    # Plot the data points of class 1 in green
    plt.plot(x_f[y_train == 1], y_f[y_train == 1], z_f[y_train == 1], "og")
    # Get the coefficients of the decision boundary (the plane)
    thetaN = lr.coef_
    theta0 = lr.intercept_
    theta1 = thetaN[0][0]
    theta2 = thetaN[0][1]
    theta3 = thetaN[0][2]
    # Define the x and y ranges of the plane
    x = numpy.array([-0.9, 0.9])
    y = numpy.arange(-0.9, 0.9)
    # Create a meshgrid for x and y
    x, y = numpy.meshgrid(x, y)
    # Calculate the z values of the plane
    z = -(theta0+theta1*x+theta2*y)/(theta3)
    # Plot the decision boundary (the plane)
    plt.gca().plot_surface(x, y, z, shade=False, color='y')
    # Show the plot
    plt.show()

# Function to run the classifiers and print their accuracies
def run_classifier(x_train, y_train, x_test, y_test):
    # Fit a Logistic Regression model to the training data
    lr = fit_LR(x_train, y_train)
    # Predict the labels of the test data and calculate the accuracy
    acc, conf_matrix = predict_and_score(lr, x_test, y_test)
    # Print the accuracy of the Logistic Regression model
    print("LR: \t", acc)

    # Fit a Bernoulli Naive Bayes model to the training data
    nb = fit_Bernoulli_NB(x_train, y_train)
    # Predict the labels of the test data and calculate the accuracy
    acc, conf_matrix = predict_and_score(nb, x_test, y_test)
    # Print the accuracy of the Bernoulli Naive Bayes model
    print("NB: \t", acc)

    # Fit a Support Vector Machine (SVM) model to the training data
    svm = fit_SVM(x_train, y_train, _gamma="scale")
    # Predict the labels of the test data and calculate the accuracy
    acc, conf_matrix = predict_and_score(svm, x_test, y_test)
    # Print the accuracy of the SVM model
    print("SVM: \t", acc)

    # Fit a K-Nearest Neighbors (KNN) model to the training data
    _knn = fit_KNN(x_train, y_train, _algorithm="ball_tree", _weights="distance")
    # Predict the labels of the test data and calculate the accuracy
    acc, conf_matrix = predict_and_score(_knn, x_test, y_test)
    # Print the accuracy of the KNN model
    print("KNN: \t", acc)

    # Save the trained models for future use
    save_model(lr, "models_trained/lr_model")
    save_model(nb, "models_trained/nb_model")
    save_model(svm, "models_trained/svm_model")
    save_model(_knn, "models_trained/knn_model")

# Function to save a trained model to a file
def save_model(model, filename):
    # Add the .sav extension to the filename
    _filename = filename+'.sav'
    # Save the model to the file
    pickle.dump(model, open(_filename, 'wb'))

# Function to load a trained model from a file
def load_model(filename):
    # Load the model from the file
    loaded_model = pickle.load(open(filename, 'rb'))
    # Return the loaded model
    return loaded_model