import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file
filename = 'data/audio-clips/sample-1.wav'
y, sr = librosa.load(filename)

# Plot the waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Compute the mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert to decibel units
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
