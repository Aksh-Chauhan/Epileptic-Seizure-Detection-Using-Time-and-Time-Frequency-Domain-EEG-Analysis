# Epileptic-Seizure-Detection-Using-Time-and-Time-Frequency-Domain-EEG-Analysis

An end-to-end deep learning framework designed to detect epileptic seizures from EEG signals using the **Bonn EEG Dataset**. This model combines a dual-branch architecture—extracting raw **Time-Domain** features via 1D-CNN and **Time-Frequency Domain** features via Short-Time Fourier Transform (STFT) with 2D-CNN—before feeding the fused representations into a Bidirectional LSTM (BiLSTM) for temporal sequence modeling.

## Model Architecture

<img width="1643" height="586" alt="image" src="https://github.com/user-attachments/assets/c9a49b9c-b8fe-4841-8a80-bb4d1d17f386" />

The network leverages multi-scale feature extraction:

**1D CNN Branch:** Extracts fine temporal dynamics directly from normalized raw waveforms.

**2D CNN Branch:** Captures spectral power variations across time using magnitude spectrograms from Short-Time Fourier Transform (STFT).

**BiLSTM Backbone:** Fuses both representations and feeds them into two stacked Bidirectional LSTM layers to capture long-term bi-directional contextual dependencies.

## Experimental Cases Overview

###Case 1: Binary Seizure Detection (Seizure vs. Non-Seizure)

Case 1 focuses on standard automated seizure detection by isolating active ictal (seizure) events from all non-seizure EEG recordings.

Class Mapping:

Class 0 (Non-Seizure): Sets Z, O, N, and F (Healthy subjects and inter-ictal recordings combined).

Class 1 (Seizure): Set S (Ictal seizure recordings).

