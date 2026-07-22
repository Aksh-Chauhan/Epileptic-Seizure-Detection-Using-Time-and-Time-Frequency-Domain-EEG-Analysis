# Epileptic-Seizure-Detection-Using-Time-and-Time-Frequency-Domain-EEG-Analysis

An end-to-end deep learning framework designed to detect epileptic seizures from EEG signals using the **Bonn EEG Dataset**. This model combines a dual-branch architecture—extracting raw **Time-Domain** features via 1D-CNN and **Time-Frequency Domain** features via Short-Time Fourier Transform (STFT) with 2D-CNN—before feeding the fused representations into a Bidirectional LSTM (BiLSTM) for temporal sequence modeling.

## Model Architecture

<img width="1643" height="586" alt="image" src="https://github.com/user-attachments/assets/c9a49b9c-b8fe-4841-8a80-bb4d1d17f386" />

The network leverages multi-scale feature extraction:

**1D CNN Branch:** Extracts fine temporal dynamics directly from normalized raw waveforms.

**2D CNN Branch:** Captures spectral power variations across time using magnitude spectrograms from Short-Time Fourier Transform (STFT).

**BiLSTM Backbone:** Fuses both representations and feeds them into two stacked Bidirectional LSTM layers to capture long-term bi-directional contextual dependencies.

## Experimental Cases Overview

**Case 1: Binary Seizure Detection (Seizure vs. Non-Seizure)**

- Case 1 focuses on standard automated seizure detection by isolating active ictal (seizure) events from all non-seizure EEG recordings.

- Target Problem: Binary classification determining whether a given EEG segment contains active seizure activity.

- Class Mapping:

  - Class 0 (Non-Seizure): Sets Z, O, N, and F (Healthy subjects and inter-ictal recordings combined).

  - Class 1 (Seizure): Set S (Ictal seizure recordings).

**Case 2: 3-Class Brain State Categorization**
- Case 2 groups the five dataset subsets into three distinct physiological states, allowing the model to differentiate between healthy baseline activity, inter-ictal (seizure-free intervals), and ictal (active seizure) events.

- Target Problem: Multi-class classification across three clinical condition categories.

- Class Mapping:

  - Class 0 (Healthy Control): Set Z (Surface EEG, eyes open).

  - Class 1 (Baseline / Inter-ictal): Sets O (Surface EEG, eyes closed) and N (Seizure-free inter-ictal recordings from the hippocampal structure).

  - Class 2 (Focal Inter-ictal / Ictal): Sets F (Seizure-free inter-ictal recordings from the epileptogenic zone) and S (Ictal seizure recordings).

**Case 3: 5-Class Granular Signal Identification**

- Case 3 evaluates the dual-branch model's capacity for fine-grained discrimination by assigning every original Bonn subset (Z, O, N, F, S) to its own distinct label.

- Target Problem: Full 5-way multi-class classification mapping every distinct recording condition.

- Class Mapping:

  - Class 0: Set Z — Healthy subject, Surface EEG (Eyes Open)

  - Class 1: Set O — Healthy subject, Surface EEG (Eyes Closed)

  - Class 2: Set N — Epileptic patient, Intracranial EEG (Inter-ictal, Hippocampal structure)

  - Class 3: Set F — Epileptic patient, Intracranial EEG (Inter-ictal, Epileptogenic zone)

  - Class 4: Set S — Epileptic patient, Intracranial EEG (Ictal, Active seizure)

## Performance & Metrics

Each script automatically evaluates model performance on an 80/20 train/validation split and logs key metrics:

**Accuracy:** Overall classification accuracy.

**F1-Score:** Weighted harmonic mean of precision and recall.

**Confusion Matrix:** Complete breakdown of true positive vs. misclassified instances across all target classes.

**Sensitivity & Specificity:** Calculated directly for Case 1 binary detection.

**Diagnostic Plots:** Training vs. Validation Loss and Accuracy curves generated post-training.
