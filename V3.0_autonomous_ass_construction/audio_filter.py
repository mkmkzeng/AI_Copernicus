import os
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np

# Define the high-pass filter to remove background noise
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_highpass_filter(audio_path, output_path, cutoff=300, order=5):
    # Load the audio file
    sample_rate, audio_data = wavfile.read(audio_path)

    # Normalize the audio data if it's in int format
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        audio_data = audio_data / np.max(np.abs(audio_data))

    # Apply the high-pass filter
    b, a = butter_highpass(cutoff, sample_rate, order)
    filtered_audio = signal.filtfilt(b, a, audio_data)

    # Avoid clipping
    filtered_audio = np.clip(filtered_audio, -1, 1)

    # Save the filtered audio to a new file
    wavfile.write(output_path, sample_rate, (filtered_audio * 32767).astype(np.int16))

def batch_process(input_folder, output_folder, cutoff=300, order=5):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all .wav files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing {filename}...")

            # Apply high-pass filter to each file
            apply_highpass_filter(input_path, output_path, cutoff, order)

# Batch process all .wav files
input_folder = "./dataset/raw_audio"
output_folder = "./dataset/pure_audio"
batch_process(input_folder, output_folder)

