from random import seed
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def imp_dataset(dataset_path):
    # Load the dataset from the provided path
    df_voice = pd.read_csv(dataset_path)

    # Shuffle the dataset to ensure randomness
    df_voice = shuffle(df_voice, random_state=42)

    # Replace the string labels with numerical labels for easier processing
    # male -> 0 and female -> 1
    df_voice["label"] = df_voice["label"].replace("male", 0)
    df_voice["label"] = df_voice["label"].replace("female", 1)

    # Get all column names except "label" (these are the features)
    features = df_voice.keys()
    features = features.drop("label")  # remove label

    # Split the dataset into training and test sets
    # 80% of the data is used for training and 20% for testing
    df1 = df_voice.iloc[: int(len(df_voice)*0.8)]  # 80% of data
    df2 = df_voice.iloc[int(len(df_voice)*0.8)+1:]  # 20% of data

    # Create the training set
    x_train = df1.loc[:, features].values
    y_train = df1.loc[:, ['label']].values

    # Create the test set
    x_test = df2.loc[:, features].values
    y_test = df2.loc[:, ['label']].values

    # Standardize the features to have mean=0 and variance=1 for better performance of machine learning algorithms
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    # Return the training and test sets
    return x_train, y_train, x_test, y_test