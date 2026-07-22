# Epileptic-Seizure-Detection-Using-Time-and-Time-Frequency-Domain-EEG-Analysis

An end-to-end deep learning framework designed to detect epileptic seizures from EEG signals using the **Bonn EEG Dataset**. This model combines a dual-branch architecture—extracting raw **Time-Domain** features via 1D-CNN and **Time-Frequency Domain** features via Short-Time Fourier Transform (STFT) with 2D-CNN—before feeding the fused representations into a Bidirectional LSTM (BiLSTM) for temporal sequence modeling.

## Model Architecture

<img width="1643" height="586" alt="image" src="https://github.com/user-attachments/assets/c9a49b9c-b8fe-4841-8a80-bb4d1d17f386" />

The network leverages multi-scale feature extraction:

**1D CNN Branch:** Captures high-resolution raw temporal dynamics.

**2D CNN Branch:** Captures spectral patterns and frequency shifts across time using STFT magnitude spectrograms.

**BiLSTM Backbone:** Captures bi-directional context from the combined feature maps to classify seizure vs. non-seizure activity.
