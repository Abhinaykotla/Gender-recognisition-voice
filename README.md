# Gender-recognisition-voice

This project uses multiple machine learning models to recognize the gender of a speaker from a .wav audio file. It analyzes 14 individual traits of human voice to make the prediction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires Python 3 and several Python libraries, including:

- scikit-image
- pandas
- numpy
- matplotlib
- pyAudioAnalysis
- scipy
- sklearn
- hmmlearn
- simplejson
- eyed3
- pydub

You can install these requirements using pip:

```bash
pip3 install -r requirements.txt


## Usage

Running the Classifiers
To run the classifiers, use the following command:

```
python3 main.py -r
```

To run the classifier with a new audio sample (in .wav format), use:

```
python3 main.py -w path/file.wav
```

Note: If you need to convert an audio file into .wav format, use [SoundConverter](https://soundconverter.org/).

Classifying a New CSV Sample
To run the classifier with a new sample in CSV format, use:
```
python3 main.py -i path/file.csv
```