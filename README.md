# Epileptic-Seizure-Detection-Using-Time-and-Time-Frequency-Domain-EEG-Analysis

An end-to-end deep learning framework designed to detect epileptic seizures from EEG signals using the **Bonn EEG Dataset**. This model combines a dual-branch architecture—extracting raw **Time-Domain** features via 1D-CNN and **Time-Frequency Domain** features via Short-Time Fourier Transform (STFT) with 2D-CNN—before feeding the fused representations into a Bidirectional LSTM (BiLSTM) for temporal sequence modeling.
