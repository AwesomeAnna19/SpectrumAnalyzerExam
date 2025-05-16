# Here are all the packages/libraries that I use!
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import get_window
import sounddevice as sd


# Here are ALL the parameters that you can change!
# You select a WAV file you want to input to the program (audio_file_path). Next the sampling rate of your choice (sampling_rate).
# Then choose how many samples we want the FFT to process (fft_size). Next the type of window of your choice (window_type).
# Then the length of the window (window_length), and how much the window should overlap "each other" (window_overlap_size).
# Lastly, to make the dB scaling logarithmic and not linear, you make it TRUE (use_db_scaling).
audio_file_path = 'Anna Rose - drums experiment~ 2025-02-02 21_13.wav'
sampling_rate = 48000
fft_size = 16384
window_type = 'hann'  # hann, hamming, blackman, boxcar, etc...
window_length = 4096  # 2048, 4096...
window_overlap_size = window_length // 2
use_db_scaling = True  # either linear (if FALSE) or logarithmic (if TRUE)


# Here the file will be read, where I take its own sampling rate (original_sampling_rate),
# so how many samples there are in the file, and then each sample's value (samples_values).
# Underneath, the values of each sample (samples_values) will be converted from stereo (two channels) to mono (one combined channel),
# and then the values of each sample are being normalized, that means they will be more "close to each other's values",
# so there will not be as many sudden peaks throughout the file.
original_sampling_rate, samples_values = wavfile.read(audio_file_path)
if samples_values.ndim == 2:
    samples_values = samples_values.mean(axis=1)
samples_values = samples_values / np.max(np.abs(samples_values))


# Here we find out what the duration of the file is in seconds (duration_in_seconds),
# and underneath we make an axis just for the duration in seconds (time_axis).
duration_in_seconds = len(samples_values) / sampling_rate
time_axis = np.linspace(0, duration_in_seconds, len(samples_values))


# Here we plot the modified file in a time domain representation.
# First, we choose the figure's size.
# Then we input the axis for the duration (time_axis) along with the values of each sample (samples_values).
# Secondly, we give the figure a title and label names for each axis, along with other parameters for how the figure will look.
plt.figure(figsize=(12, 4))
plt.plot(time_axis, samples_values)
plt.title("Audio File in Time Domain Representation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()


# Here we actually compute the Fast Fourier Transformation (FFT) (fft_results), with using the NumPy library.
# Then we store the FFT values, and convert those values in a dB magnitude spectrum (magnitude_spectrum).
# We also check if the dB scaling for the magnitude spectrum (use_db_scaling) is TRUE, if TRUE it will go through the if statement and become logarithmic,
# and if FALSE it will NOT go through the if statement and therefore become linear.
# Lastly we take the dB values from the FFT values, convert them to frequencies in Hz, and make an axis for that (frequency_axis).
fft_results = np.fft.fft(samples_values, n=fft_size)
magnitude_spectrum = np.abs(fft_results)[:fft_size // 2]
if use_db_scaling:
    magnitude_spectrum = 20 * np.log10(magnitude_spectrum + 1e-10)
frequency_axis = np.fft.fftfreq(fft_size, d=1/sampling_rate)[:fft_size // 2]


# Here we plot the dB values from the FFT values in a frequency domain representation.
# First, we choose the figure's size.
# Then we input the axis for the frequencies in Hz along with the dB values from the FFT values.
# Secondly, we give the figure a title and label names for each axis, along with other parameters for how the figure will look.
plt.figure(figsize=(12, 4))
plt.plot(frequency_axis, magnitude_spectrum)
plt.xlim(0, 5000)
plt.title("Processed FFT on Audio File in Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude ({})".format("dB" if use_db_scaling else "Linear"))
plt.grid(True)
plt.tight_layout()
plt.show()


# Here we make the spectrogram of the original file that is mono and has been normalized,
# along with its sampling rate, the type of window, the length of the window,
# and lastly how much the window should overlap "each other".
# The spectrogram will output all its frequencies, time points, and then its spectral density,
# which estimates the power/energy at a specific time and frequency in the file.
frequencies, times, spectral_density = spectrogram(
    samples_values,
    fs=sampling_rate,
    window=window_type,
    nperseg=window_length,
    noverlap=window_overlap_size
)


# Here we check if the dB scaling for the spectrogram is TRUE (use_db_scaling),
# if it is TRUE then it will go through the if statement and become logarithmic,
# and if it is FALSE then it will NOT go through the if statement and therefore become linear.
if use_db_scaling:
    spectral_density = 10 * np.log10(spectral_density + 1e-10)


# Here we plot the spectrogram of the original file that is mono and has been normalized.
# First, we choose the figure's size.
# Then we input the axis for the frequency in Hz and the duration in seconds.
# Next we make a colorbar that shows the dB scaling, if it is logarithmic.
# Secondly, we give the figure a title and label names for each axis, along with other parameters for how the figure will look.
plt.figure(figsize=(12, 5))
plt.pcolormesh(times, frequencies, spectral_density, shading='gouraud', cmap='jet')
plt.ylim(0, 5000)
plt.title("Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label='Power ({})'.format("dB" if use_db_scaling else "Linear"))
plt.tight_layout()
plt.show()


# Here we make, calculate and plot a long-term spectrum.
# A long-term spectrum shows the average level of the frequency components for the entire file.
# First we split the file into overlapping segments (step_size),
# then we calculate how many segments will fit throughout the whole file (num_segments).
# Then we make a window, in this case a Hann window, along with its length (long_term_window).
# Lastly, we gather all the magnitudes of the frequency components over all the segments (spectrum_accumulator).
step_size = window_length - window_overlap_size
num_segments = (len(samples_values) - window_length) // step_size + 1
long_term_window = get_window(window_type, window_length)
spectrum_accumulator = np.zeros(fft_size // 2)


# Here we use, apply and compute different things on each segment.
# First, we go through each segment (the for loop itself).
# Then we take a segment and apply the window we made earlier, and repeat that (window_on_segment).
# Next we compute FFT on each segment, where we make all the negative values to positive (segment_fft).
# Then we "slice" the half of the FFT results, so we only keep the upper ones, or the non-negative frequencies,
# and now we also know the strength of each frequency (magnitude_strength).
# Lastly, we sum all the strengths of each frequency across all the segments (the last line).
for i in range(num_segments):
    start = i * step_size
    window_on_segment = samples_values[start:start + window_length] * long_term_window
    segment_fft = np.fft.fft(window_on_segment, n=fft_size)
    magnitude_strength = np.abs(segment_fft[:fft_size // 2])
    spectrum_accumulator += magnitude_strength


# Here is where we calculate and compute the long-term spectrum,
# so we find out what the average level of the frequency components is, throughout the whole file.
# Then we again check if the dB scaling for the long-term spectrum is TRUE (use_db_scaling).
# If TRUE it will go through the if statement and become logarithmic,
# and if FALSE it will NOT go through the if statement and therefore become linear.
average_spectrum = spectrum_accumulator / num_segments
if use_db_scaling:
    average_spectrum = 20 * np.log10(average_spectrum + 1e-10)


# Here we plot the long-term spectrum of the entire file.
# First, we choose the figure's size.
# Then we input the axis for the frequency in Hz and the average level of the frequency components.
# Secondly, we give the figure a title and label names for each axis, along with other parameters for how the figure will look.
plt.figure(figsize=(12, 4))
plt.plot(frequency_axis, average_spectrum)
plt.xlim(0, 5000)
plt.title("Long-Term Average Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [{}]".format("dB" if use_db_scaling else "Linear"))
plt.grid(True)
plt.tight_layout()
plt.show()

# Here we can play the original file that is mono and has been normalized.
sd.play(samples_values, samplerate=sampling_rate)
sd.wait()
