import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from scipy.stats import gmean
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from functions.import_dataset import imp_dataset


def spectral_properties(file):
    # Read the audio file
    [sample_rate, data] = audioBasicIO.readAudioFile(file)
    # Extract the short-term feature sequences for the audio file
    F, f_names = audioFeatureExtraction.stFeatureExtraction(data, sample_rate, 0.050*sample_rate, 0.025*sample_rate)

    # Compute the one-dimensional n-point discrete Fourier Transform for real input
    spec = np.abs(np.fft.rfft(data))
    # Compute the one-dimensional n-point discrete Fourier Transform of real input and return the frequencies
    freq = np.fft.rfftfreq(len(data), d=1 / sample_rate)
    # Find the index of the maximum frequency (unused in this code)
    peakf = np.argmax(freq)
    # Normalize the spectrum
    amp = spec / spec.sum()
    # Calculate the mean frequency
    mean = (freq * amp).sum()
    # Calculate the standard deviation of the frequency
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    # Calculate the cumulative sum of the normalized spectrum
    amp_cumsum = np.cumsum(amp)
    # Calculate the median frequency
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    # Calculate the mode frequency (frequency with the highest amplitude)
    mode = freq[amp.argmax()]
    # Calculate the first quartile frequency
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    # Calculate the third quartile frequency
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    # Calculate the interquartile range of the frequency
    IQR = Q75 - Q25
    # Calculate the deviation of the amplitude from its mean
    z = amp - amp.mean()
    # Calculate the standard deviation of the amplitude
    w = amp.std()
    # Calculate skewness of the spectrum
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    # Calculate kurtosis of the spectrum
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    # Calculate spectral flatness
    spec_flatness = gmean(spec**2)/np.mean(spec**2)

    # Create a dictionary to store the results
    result_d = {
        'meanfreq': mean/1000,  # Mean frequency (in kHz)
        'sd': sd/1000,  # Standard deviation (in kHz)
        'median': median/1000,  # Median frequency (in kHz)
        'Q25': Q25/1000,  # First quartile (in kHz)
        'Q75': Q75/1000,  # Third quartile (in kHz)
        'IQR': IQR/1000,  # Interquartile range (in kHz)
        'skew': skew,  # Skewness
        'kurt': kurt,  # Kurtosis
        'sp.ent': F[5].mean(),  # Spectral entropy
        'sfm': spec_flatness,  # Spectral flatness
        'mode': mode/1000,  # Mode frequency (in kHz)
        'centroid': F[3].mean()/1000,  # Spectral centroid (in kHz)
    }
    # Return the results
    return result_d


def test_new_sample(file):
    # Initialize an empty list for the new sample
    new_sample = []
    # Get the spectral properties of the file
    test = spectral_properties(file)
    # Create a list of the spectral properties
    new_sample = [test[t] for t in test]

    # Normalize the new sample
    norm = Normalizer(norm='l2')
    new_sample = norm.transform(np.float64([new_sample]))
    # Import the dataset
    x_train, y_train, x_test, y_test = imp_dataset('dataset/voice.csv')
    # Initialize PCA
    pca = PCA()
    # Fit PCA on the training data
    pca.fit(x_train)
    # Transform the new sample using the fitted PCA
    new_sample = pca.transform(new_sample)[0]

    # Predict the class of the new sample using SVM
    print(svm.predict([new_sample]))
    # Return the transformed new sample
    return new_sample