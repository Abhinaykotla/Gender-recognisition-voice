# Import necessary libraries and modules
import argparse
import os
import subprocess
from functions.import_dataset import *
from functions.new_sample import *
from LR import *
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from shutil import copyfile

# Initialize argument parser
parser = argparse.ArgumentParser(description='speech gender classification')

# Define command line arguments
parser.add_argument("-w", "--wav", action="store", dest="wav", help="Take a sample wav file and classify it", type=str)
parser.add_argument("-i", "--input", action="store", dest="inp", help="Take a sample csv file and classify it", type=str)
parser.add_argument("-r", "--run", action="store_true", help="Run the classifier and see the accuracy results")
args = parser.parse_args()


# If --run or -r argument is passed, run the classifier
if args.run:
    # Import the dataset
    x_train, y_train, x_test, y_test = imp_dataset("dataset/voice.csv")
    # Normalize the data
    x_train, x_test = normalize_L2(x_train, x_test)
    # Apply PCA to the data
    x_train, x_test, pca = PCA_decomposition(x_train, x_test)
    # Save the PCA model
    pickle.dump(pca, open("models_trained/pca.sav", 'wb'))
    # Run the classifier
    run_classifier(x_train, y_train, x_test, y_test)

# If --input or --wav argument is passed, classify the input
if args.inp or args.wav:
        # Try to load the trained models
    try:
        # Load the SVM model
        svm = load_model("models_trained/svm_model.sav")
        # Load the Logistic Regression model
        lr = load_model("models_trained/lr_model.sav")
        # Load the Naive Bayes model
        nb = load_model("models_trained/nb_model.sav")
        # Load the K-Nearest Neighbors model
        _knn = load_model("models_trained/knn_model.sav")
        # Load the PCA model
        pca = pickle.load(open("models_trained/pca.sav", 'rb'))
    except:
        # If models are not found, train them
        # Import the dataset
        x_train, y_train, x_test, y_test = imp_dataset("dataset/voice.csv")
        # Normalize the data
        x_train, x_test = normalize_L2(x_train, x_test)
        # Apply PCA to the data
        x_train, x_test, pca = PCA_decomposition(x_train, x_test)
        # Save the PCA model
        pickle.dump(pca, open("models_trained/pca.sav", 'wb'))
        # Train the SVM model
        svm = fit_SVM(x_train, y_train, _gamma="scale")
        # Train the Logistic Regression model
        lr = fit_LR(x_train, y_train)
        # Train the Naive Bayes model
        nb = fit_Bernoulli_NB(x_train, y_train)
        # Train the K-Nearest Neighbors model
        _knn = fit_KNN(x_train, y_train, _algorithm="ball_tree", _weights="distance")


# If --input argument is passed, classify the input csv file
if args.inp:
    # Get the file path from the argument
    file_path = args.inp
    # Read the file and split it into lines
    file_lines = open(file_path, "r").read().split("\n")
    # Remove the first line (header)
    del file_lines[0]
    # Remove empty lines
    file_lines.remove('')
    # Initialize an empty list for the samples
    sample_csv = []
    # For each line in the file
    for f in file_lines:
        # Replace "female" with '1' and "male" with '0'
        f = f.replace("female", '1')
        f = f.replace("male", '0')
        # Remove quotes
        f = f.replace('"', '')
        # Split the line into fields and add it to the samples list
        sample_csv.append(f.split(","))

    # Define the models
    models = ["SVM", "LR", "NB", "KNN"]

    # Initialize empty lists for the samples and their labels
    x_samples = []
    y_samples = []
    # For each sample in the csv file
    for i in range(len(sample_csv)):
        # Add the features to the samples list
        x_samples.append(sample_csv[i][0:-1])
        # Add the label to the labels list
        y_samples.append(sample_csv[i][-1])

    # Normalize the samples
    norm = Normalizer(norm='l2')
    x_samples = norm.transform(x_samples)
    # Apply PCA to the samples
    x_samples = pca.transform(x_samples)

    # Predict the labels using the models
    svm_res = svm.predict(x_samples),
    lr_res = lr.predict(np.float64(x_samples)),
    nb_res = nb.predict(np.float64(x_samples)),
    _knn_res = _knn.predict(np.float64(x_samples)),

    # Initialize a list to count the successful predictions
    success = [0, 0, 0, 0]
    # Get the total number of samples
    tot = len(svm_res[0])

    # Print the predictions and the true labels
    print("SVM \t LR \t NB \t KNN \t label")
    for i in range(tot):
        print(str(svm_res[0][i])+" \t "+str(lr_res[0][i])+" \t "+str(nb_res[0][i])+" \t "+str(_knn_res[0][i])+" \t "+y_samples[i])

        # Count the successful predictions
        success[0] += 1 if int(svm_res[0][i]) == int(y_samples[i]) else 0
        success[1] += 1 if int(lr_res[0][i]) == int(y_samples[i]) else 0
        success[2] += 1 if int(nb_res[0][i]) == int(y_samples[i]) else 0
        success[3] += 1 if int(_knn_res[0][i]) == int(y_samples[i]) else 0

    # Print the number of successful predictions for each model
    print(str(success[0])+"/"+str(tot)+" \t " + str(success[1])+"/"+str(tot)+" \t " + str(success[2])+"/"+str(tot)+" \t " + str(success[3])+"/"+str(tot))

# If --wav argument is passed, classify the input wav file
if args.wav:
    # Copy the wav file to a specific location
    copyfile(args.wav, "voice.wav")

    # Suppress the output of the subprocess call
    FNULL = open(os.devnull, 'w')
    # Call the R script to extract features from the wav file
    subprocess.call(('Rscript', "R/extract_single.r"), stdout=FNULL, stderr=subprocess.STDOUT)

    # Read the second line of the csv file (excluding the header)
    # This line contains the features of the wav file
    sample = open("my_voice.csv", "r").read().split("\n")[1].split(",")
    # Convert the sample to a list of lists
    sample = [sample]

    # Normalize the sample
    norm = Normalizer(norm='l2')
    sample = norm.transform(np.float64(sample))
    # Apply PCA to the sample
    sample = pca.transform(np.float64(sample))

    # Print the labels for the predictions
    print("male: 0, female: 1")
    # Predict the label using the SVM model and print it
    print("SVM: \t", svm.predict(sample)[0])
    # Predict the label using the Logistic Regression model and print it
    print("LR: \t", lr.predict(np.float64(sample))[0])
    # Predict the label using the Naive Bayes model and print it
    print("NB: \t", nb.predict(np.float64(sample))[0])
    # Predict the label using the K-Nearest Neighbors model and print it
    print("KNN: \t", _knn.predict(np.float64(sample))[0])
