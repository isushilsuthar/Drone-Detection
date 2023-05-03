import os
import librosa
import math
import json

DATASET_PATH = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/Recorded Audios/Real World Testing"  # Path of folder with training audios.
JSON_PATH = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/files/mel_data.json"  # Location and file name to save feature extracted data.

SAMPLE_RATE = 22050  # Sample rate in Hz.
DURATION = 10  # Length of audio files fed. Measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mels=90, n_fft=2048, hop_length=512, num_segments=5):
    # num_segments let's you chop up track into different segments to create a bigger dataset.
    # Value is changed at the bottom of the script.

    # Dictionary to store data into JSON_PATH
    data = {
        "mapping": [],  # Used to map labels (0 and 1) to category name (UAV and no UAV).
        "mel": [],  # Mels are the training input, labels are the target.
        "labels": []  # Features are mapped to a label (0 or 1).
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mel_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Loops through all the folders in the training audio folder.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensures that we're not at the root level.
        if dirpath is not dataset_path:

            # Saves the semantic label for the mapping.
            dirpath_components = dirpath.split("/")  # class/background => ["class", "background"]
            semantic_label = dirpath_components[-1]  # considering only the last value
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Processes all the audio files for a specific class.
            for f in filenames:

                # Loads audio file.
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Process segments, extracting mels and storing data to JSON_PATH.
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment  # s=0 --> num_samples_per_segment
                    y = signal[start_sample:finish_sample]
                    mel = librosa.feature.melspectrogram(y= signal[start_sample:finish_sample],
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         n_mels=n_mels,
                                                         hop_length=hop_length)
                    db_mel = librosa.power_to_db(mel)
                    mel = db_mel.T
                    # Stores mels for segment, if it has the expected length.
                    if len(mel) == expected_num_mel_vectors_per_segment:
                        data["mel"].append(mel.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    # num_segments can be changed. 10 with 10 second audio equates to a segment equalling 1 second.



"""
This code extracts Mel Frequency Cepstral Coefficients (MFCC) from audio files and saves them in a JSON file. MFCC are commonly used for speech recognition and audio classification tasks. The extracted features are used to train machine learning models for acoustic UAV (Unmanned Aerial Vehicle) identification.

The code loops through all the subdirectories in a given directory and loads all the audio files it finds. For each audio file, it divides it into segments of a specified duration and extracts the MFCC for each segment. It then stores the extracted features in a dictionary along with the corresponding labels (0 for no UAV, 1 for UAV). This dictionary is then written to a JSON file.

The parameters used for feature extraction can be customized, such as the number of Mel bands (n_mels), the length of the FFT window (n_fft), the hop length (hop_length), and the number of segments to divide the audio file into (num_segments). The path to the audio files directory and the path to the output JSON file can also be customized.
"""
'''
Function: save_mfcc

Parameters:

dataset_path: Path of folder with training audios.
json_path: Location and file name to save feature extracted data.
n_mels (optional): Number of Mel frequency bins to generate.
n_fft (optional): Length of the FFT window.
hop_length (optional): Number of samples between successive frames.
num_segments (optional): Number of segments to divide the audio file into.
Default values:

n_mels = 90
n_fft = 2048
hop_length = 512
num_segments = 5
Variables:

DATASET_PATH: Path of folder with training audios.
JSON_PATH: Location and file name to save feature extracted data.
SAMPLE_RATE: Sample rate in Hz.
DURATION: Length of audio files fed. Measured in seconds.
SAMPLES_PER_TRACK: SAMPLE_RATE * DURATION
Output:

Saves a dictionary of data to the specified JSON_PATH. The dictionary contains:
"mapping": Used to map labels (0 and 1) to category name (UAV and no UAV).
"mel": Mels are the training input, labels are the target.
"labels": Features are mapped to a label (0 or 1).
'''