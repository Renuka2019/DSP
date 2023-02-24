import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load audio file
rate, audio = wav.read("speech.wav")

# Normalize audio data
audio = audio / np.max(np.abs(audio))

# Apply pre-emphasis filter
pre_emphasis = 0.97
emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

# Define frame length and overlap
frame_length = int(0.025 * rate)
frame_overlap = int(0.01 * rate)

# Generate frames of audio data
frames = []
for i in range(0, len(emphasized_audio) - frame_length, frame_overlap):
    frame = emphasized_audio[i:i + frame_length]
    frames.append(frame)

# Apply Hamming window to frames
windowed_frames = np.array(frames) * signal.hamming(frame_length)

# Compute power spectrum of frames using Fourier Transform
power_spectrum = np.abs(np.fft.rfft(windowed_frames, n=512))**2

# Compute Mel filterbank energies
num_filters = 26
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
hz_points = 700 * (10**(mel_points / 2595) - 1)  # Convert Mel to Hz
bin = np.floor((frame_length + 1) * hz_points / rate)

filterbank = np.zeros((num_filters, int(np.floor(frame_length / 2 + 1))))
for m in range(1, num_filters + 1):
    f_m_minus = int(bin[m - 1])   
    f_m = int(bin[m])
    f_m_plus = int(bin[m + 1])  

    for k in range(f_m_minus, f_m):
        filterbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        filterbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filterbank_energies = np.dot(power_spectrum, filterbank.T)
filterbank_energies = np.where(filterbank_energies == 0, np.finfo(float).eps, filterbank_energies)  # Numerical Stability

# Compute logarithmic Mel spectrum
log_filterbank_energies = 20 * np.log10(filterbank_energies)

# Apply discrete cosine transform (DCT) to decorrelate the filterbank energies
num_ceps = 12
mfcc = np.zeros((len(log_filterbank_energies), num_ceps))
for i in range(num_ceps):
    norm = np.sqrt(2.0 / frame_length)
    if i == 0:
        norm = np.sqrt(1.0 / frame_length)
    for j in range(len(log_filterbank_energies)):
        mfcc[j, i] = norm * np.sum(log_filterbank_energies[j] * np.cos(np.pi * i / num_filters * (np.arange(num_filters) + 0.5)))

# Use the first 12 MFCC coefficients as features for speech recognition
features = mfcc[:, 0:12]

# Train a
