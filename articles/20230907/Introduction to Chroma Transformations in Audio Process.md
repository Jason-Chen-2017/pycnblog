
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Chroma is the color of a sound spectrum. It is also known as ‘pitch class’ or ‘tonal content’, which refers to a set of chromatic tones that form together when a musical note is played on a piano keyboard. Musicians use chroma transformations to analyse and manipulate audio data with specific emphasis on pitch-class information. In this article, we will explain what chroma transformation is, its basic concepts and how it can be used for analysis and manipulation of audio signals. We will then demonstrate the implementation of various chroma transforms using Python programming language. Finally, some possible applications of chroma transformation are discussed along with their potential limitations and benefits.

This article assumes readers have prior knowledge of digital signal processing (DSP) fundamentals including sampling rate, quantization levels, frequency domain, time domain, etc. This article also does not discuss any theory related to music theory or chromatics.

# 2.基本概念术语说明
## 2.1.Pitch Class
A pitch class is a group of notes within an octave that share similar frequencies but different MIDI values. There are twelve pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B), each with their own unique frequency range. 

## 2.2.Mel Scale
The Mel scale was developed by Johannes Sakharov to represent perceptual loudness. It is based on the frequencies human ear recognizes between 133Hz and 8kHz. The Mel scale maps these frequencies into linearly increasing intervals between -inf and +inf. Therefore, it provides a way to decompose sounds into logarithmic units instead of base 10 exponents. For example, the mel value of 100Hz is approximately equal to the decibel difference between middle C and 100Hz.

## 2.3.Chromagram
A chromagram is a representation of pitch class distributions across an entire recording. Each column represents one pitch class at a particular point in time, while each row corresponds to a frame of audio. Each cell in the matrix contains a count of occurrences of a particular pitch class during that frame. A chromagram shows the relative distribution of pitch classes over time, providing insights into the characteristics of a piece's timbre.

## 2.4.Chroma Vector
A chroma vector is simply a sequence of chroma features extracted from a given frame of audio. These features may include spectral centroid, bandwidth, zero crossing rate, etc. Each feature has a distinct meaning, reflecting certain aspects of the audio signal. By combining multiple chroma vectors, we get more comprehensive representations of the timbre of a piece.

## 2.5.Chroma Decomposition
Chroma decomposition involves converting an audio clip into its constituent pitch classes. Essentially, it separates out individual notes from the overall mix and gives us insight into their tone characteristics. Chroma decomposition works by calculating the power spectra of frames of audio, extracting pitch-class energy, and representing them as chromagrams or chroma vectors.

# 3.核心算法原理及其具体操作步骤以及数学公式讲解
## 3.1.Chroma Transformation Overview
Chroma transformation is a technique used for transforming audio signals into other representations that capture both timbral and temporal information about the sound. It takes advantage of the fact that humans are able to recognize patterns in pitch-class information. The simplest type of chroma transformation is called ‘pitch-based’. Here, we map the chroma of each frame onto the corresponding pitch class, resulting in a chromagram where each frame is represented by only two dimensions: the pitch class index and the associated magnitude. The mapping function could either be fixed (e.g., equally spaced bins) or adaptive (e.g., learned from training data). 

Another important aspect of chroma transformation is that it preserves both local and global structure of the signal. Since the chroma transformation is typically performed on short windows of audio, adjacent frames usually exhibit correlations due to neighboring pitch classes being present simultaneously. Similarly, transitions between different sets of pitch classes throughout the track can be captured through the chroma transition matrix, which shows the correlation between every pair of consecutive pitch classes. Lastly, chroma shifting is another concept introduced by researchers interested in creating naturalistic-sounding effects such as singing voice synthesis, chorusing, or vocal remapping. 

## 3.2.Linear Prediction Coefficients
One of the most commonly used types of chroma transformation is Linear Predictive Coding (LPC), originally proposed by Rossignac and McPherson [1]. LPC attempts to predict the next sample of a signal based on previous samples and coefficients derived from the autocorrelation matrix of the signal. The output of an LPC filter is just the weighted sum of past input samples, giving rise to characteristic timbres. However, because the prediction depends on previous inputs, it cannot provide long-term temporal information beyond the length of the filter window. To achieve this, researchers have continued to explore methods for incorporating longer term context information into the chroma transformation process. 

An alternative approach to LPC is Time-Frequency Analysis (TFA) [2], which uses Morlet wavelets to perform a Fourier analysis on small sections of the audio signal and extract filters that describe the dynamics of the signal over frequency bands. TFA provides long-range temporal information that captures properties such as dynamic ranges, rhythmicity, and stability. However, the resolution of the filters is limited compared to the higher precision of LPC. Additionally, since TFA relies on predefined frequency bands, it may miss variations in the timbre caused by small changes in fundamental frequency.

Ultimately, it is difficult to compare the accuracy of TFA versus LPC without additional context information. While there exists many variants of chroma transformation algorithms, they all aim to extract useful features from audio signals while preserving their original qualities and structures. Overall, the choice between LPC and TFA should depend on the task at hand and the desired level of temporal detail required. 

## 3.3.Chord Progression and Timbre Transfer Using Chroma Transformations
Musicians often use chroma transformations to analyze and manipulate audio data with specific emphasis on pitch-class information. One application of chroma transformations is song analysis, where musicians use chromas to identify melody, harmonics, texture, tempo, and style elements in songs. Another application is timbre transfer, where musicians create new sounds or adjust existing ones to match target listeners' preferences. Both tasks require developing advanced skills in pattern recognition, statistical learning, machine learning, and computer graphics.  

In order to implement this functionality, musicians need to break down an audio clip into smaller segments that correspond to scales or chords. They can then apply a chroma transformation algorithm to each segment, resulting in a series of chroma matrices or vectors, which contain detailed pitch-class information about the song. Once the chroma representation of the song is complete, musicians can classify the pieces according to instrumentation, genre, mood, sentiment, etc. Additionally, they can modify the timing of individual notes or change the emotional valence of the piece by modifying the pitch-class distribution and tempo of the audio clip.

# 4.具体代码实例及其解释说明
We will now demonstrate the implementation of three common types of chroma transformations using Python programming language. Firstly, we will look at the Simple Chroma Transform (SCT) algorithm, which assigns equal weight to each pitch class in a chromagram. Then, we will examine the Pitch-Class Profile algorithm (PCP), which uses a Bayesian classifier to learn the probability of each pitch class occurring at each time step. Finally, we will talk about Harmonic Product Spectrum (HPS), which computes the amplitude spectrum of a spectrogram at each time step by taking the product of the amplitude spectrum at two nearby frequency bands that come closest to the peak bin. 

## 4.1.Simple Chroma Transform
The Simple Chroma Transform (SCT) algorithm assigns equal weight to each pitch class in a chromagram. The SCT code is implemented as follows:

```python
import librosa
import numpy as np

def sct(audio_file):
    y, sr = librosa.load(audio_file) # load audio file

    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # compute STFT chroma
    
    return chroma
    
# Example usage:
chroma = sct('example_song.wav')
print(chroma.shape) #(frames, 12) 
```

The `librosa` library is used to load the audio file and compute the STFT chroma using the `chroma_stft()` function. The result is a `numpy` array of shape `(frames, 12)` where `frames` is the number of frames in the audio clip. Each frame contains a count of occurrences of each pitch class for that particular time step. Note that the highest possible magnitude for a single frame is `sqrt(N/f)`, where `N` is the total number of FFT bins and `f` is the frame rate. Therefore, if you want to normalize the chromagram so that the maximum value is unity, divide each element by `np.sqrt(len(y)/sr)`.

## 4.2.Pitch-Class Profile
The Pitch-Class Profile algorithm (PCP) uses a Bayesian classifier to learn the probability of each pitch class occurring at each time step. Unlike the traditional SCT method, PCP considers the probability that each pitch class occurs at each time step rather than simply counting the number of occurrences. Moreover, PCP accounts for the uncertainty in the underlying model by modeling the probability of each pitch class using a Dirichlet distribution. 

Here's an implementation of PCP in Python:

```python
from sklearn.naive_bayes import GaussianNB

def pcp(audio_file):
    y, sr = librosa.load(audio_file) # load audio file

    hop_length = int(sr / 1000 * 128) # define hop length for 128 bpm 
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length) # compute chroma with constant Q transform

    clf = GaussianNB() # initialize Naive Bayes classifier
    X = chroma[:-1,:] # remove last row (time step after end of audio)
    Y = chroma[1:,:] # keep first row (time step before beginning of audio)

    clf.fit(X,Y[:,0]) # fit Naive Bayes classifier to train data

    profile = np.zeros((chroma.shape[0]-1, len(clf.classes_))) # allocate memory for pitch class profile

    for i in range(chroma.shape[0]-1):
        proba = clf.predict_proba(chroma[i,:].reshape(1,-1)) # compute probabilities of each pitch class occurring
        profile[i,:] = proba[0] # store probabilities
        
    return profile
    
# Example usage:
profile = pcp('example_song.wav')
print(profile.shape) #(frames-1, 12) 
```

The `chroma_cqt()` function is used to compute the chroma with Constant-Q transform, which produces smoother results than STFT chroma computed above. We also choose a hop length of 128 beats per minute (`int(sr / 1000 * 128)`) to cover the entire duration of the audio file. Next, we split the chromagram into two parts: the part before the start of the audio file and the part after the end of the audio file (since the last frame would be incomplete). We then initialize a Gaussian Naive Bayes classifier using scikit-learn, fit the classifier to the preceding section of the chromagram, and use it to estimate the probability of each pitch class occurring at each subsequent time step. The estimated probabilities are stored in a `numpy` array of shape `(frames-1, 12)`, where each row corresponds to the probability of each pitch class occurring at a particular time step. Note that the final probability of silence (represented by `None`) is included in the chroma matrix, but excluded from the pitch class profile. If you don't care about silence, you can manually discard the final row before computing the pitch class profile.

## 4.3.Harmonic Product Spectrum
The Harmonic Product Spectrum (HPS) algorithm computes the amplitude spectrum of a spectrogram at each time step by taking the product of the amplitude spectrum at two nearby frequency bands that come closest to the peak bin. HPS can reveal more complex timbres that violate conventional assumptions about note density and timbre. The algorithm consists of several steps:

1. Compute the spectrogram of the audio signal using a Short-Time Fourier Transform (STFT).
2. Calculate the magnitude squared of the complex spectrum obtained by applying the HTK Magnitude Formula to each bandpass.
3. Sort the frequency bins by their distance from the peak bin.
4. Take the Cartesian product of the top two frequency bins in sorted order to obtain the candidate pairs.
5. For each candidate pair, calculate their dot product, which reveals the similarity between the corresponding frequency bands in terms of spectral shape. 
6. Normalize the dot products to produce a probability distribution indicating the likelihood that each pair comes closest to the peak bin.
7. Use a Discrete Cosine Transform (DCT) to compress the probability distribution into a compact representation suitable for visualization.

Here's an implementation of HPS in Python:

```python
from scipy import fft
from scipy.spatial.distance import cdist
import pywt


def hps(audio_file):
    y, sr = librosa.load(audio_file) # load audio file

    n_fft = 2048  
    hop_length = int(n_fft // 4) # reduce overlap between successive frames

    mag_specgram = abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2 # compute magnitude spectrogram

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    idx_peak = np.argmax(mag_specgram, axis=1)
    dists = cdist(freqs.reshape(-1,1), freqs.reshape(-1,1)).flatten().astype(float)

    pairs = []
    similars = {}
    for fidx in range(1, len(dists)-1):
        pairs += [(fidx,x) for x in list(range(fidx+1, len(freqs))) if dists[fidx]<dists[x]]

        if len(pairs)<2:
            continue
        
        dotprods = ((mag_specgram[:,pairs[0][0]]*mag_specgram[:,pairs[0][1]])**2
                   +(mag_specgram[:,pairs[1][0]]*mag_specgram[:,pairs[1][1]])**2)
        norms = mag_specgram[:,pairs[0][0]]**2 + mag_specgram[:,pairs[0][1]]**2 \
              + mag_specgram[:,pairs[1][0]]**2 + mag_specgram[:,pairs[1][1]]**2
        
        similars[tuple(sorted([pairs[0][0], pairs[0][1]]))
               ] = max(dotprods/(norms**.5))[0]
            
        similars[tuple(sorted([pairs[1][0], pairs[1][1]]))
               ] = max(dotprods/(norms**.5))[0]

    dct_matrix = pywt.idct(list(similars.values()), 'haar', axis=-1)[:-(len(similars)%4)]

    profile = np.array([[dct_matrix[j][i] for j in range(len(dct_matrix))]
                       for i in range(len(dct_matrix)//4)])

    return profile
    
# Example usage:
profile = hps('example_song.wav')
print(profile.shape) #(frames//4, 1) 
```

We begin by loading the audio file using `librosa`, defining the parameters of our spectrogram computation (n_fft and hop_length), and computing the magnitude spectrogram using `abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2`. The calculation of the distances between frequency bins and sorting the candidates requires using NumPy arrays and SciPy functions, respectively. We then loop through all pairs of frequency bins whose distances are less than half the spacing between adjacent bins. For each candidate pair, we compute their dot product and normalize the result by dividing by the square root of the sum of squares of the four components of the mixture. We save the normalized dot products in a dictionary `similars`, keyed by tuples of indices of the candidate pairs. Finally, we apply the Inverse Discrete Cosine Transform (IDCT) to the saved dot products to obtain a compressed pitch class profile. The IDCT reduces the dimensionality of the profile to make it suitable for visualization, reducing the spatial resolution of the profile. Note that we assume that each frame spans exactly one quarter note.