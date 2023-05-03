import os
import librosa
import math
import json

DATASET_PATH = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/Recorded Audios/Real World Testing"  # Path of folder with training audios.
JSON_PATH = "C:/Users/sushi/Desktop/IIT BHU/4 sem/EE272 Explo/Acoustic-UAV-Identification/files/mfcc_data.json"  # Location and file name to save feature extracted data.

SAMPLE_RATE = 22050  # Sample rate in Hz.
DURATION = 10  # Length of audio files fed. Measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=20, n_fft=2048, hop_length=512, num_segments=5):
    # num_segments let's you chop up track into different segments to create a bigger dataset.
    # Value is changed at the bottom of the script.

    # Dictionary to store data into JSON_PATH
    data = {
        "mapping": [],  # Used to map labels (0 and 1) to category name (UAV and no UAV).
        "mfcc": [],  # MFCCs are the training input, labels are the target.
        "labels": []  # Features are mapped to a label (0 or 1).
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2

    # Loops through all the folders in the training audio folder.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensures that we're not at the root level.
        if dirpath is not dataset_path:

            # Saves the semantic label for the mapping.
            dirpath_components = dirpath.split("/")     # class/background => ["class", "background"]
            semantic_label = dirpath_components[-1]     # considering only the last value
            data["mapping"].append(semantic_label)
            print("/nProcessing {}".format(semantic_label))

            # Processes all the audio files for a specific class.
            for f in filenames:

                # Loads audio file.
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process segments, extracting mfccs and storing data to JSON_PATH.
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 --> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 --> num_samples_per_segment
                    y = signal[start_sample:finish_sample]
                    mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Stores mfccs for segment, if it has the expected length.
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    # num_segments can be changed. 10 with 10 second audio equates to a segment equalling 1 second.


"""
This code defines a function save_mfcc that extracts Mel Frequency Cepstral Coefficients (MFCCs) from audio files in a specified folder and saves the features and their corresponding labels to a JSON file. The function takes several arguments including the path to the audio files, path to the JSON file, number of MFCCs to extract, length of the FFT window, length of the hop (number of samples between each MFCC extraction), and number of segments to divide each audio file into. The number of segments determines the size of the dataset. The code loops through all the audio files in the specified folder and extracts MFCCs from each segment. The extracted features are then appended to a list of MFCCs and their corresponding labels, which is saved to the specified JSON file. Finally, the code includes a main function that calls save_mfcc and passes in the necessary arguments.
"""