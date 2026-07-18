import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling1D, MaxPooling2D, ReLU, Input, Concatenate, Bidirectional, LSTM, Flatten, Dropout, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to load and preprocess data
def load_data(base_path):
    data = []
    labels = []
    label_mapping = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1}
    for label_folder in ['Z', 'O', 'N', 'F', 'S']:
        folder_path = os.path.join(base_path, label_folder)
        for file in os.listdir(folder_path):
            if file.endswith('.txt'):
                file_path = os.path.join(folder_path, file)
                signal = np.loadtxt(file_path)
                data.append(signal)
                labels.append(label_mapping[label_folder])
    return np.array(data), np.array(labels)

# Function to apply STFT to each signal
def apply_stft(data, fs=173.61, nperseg=128):
    stft_data = []
    for signal in data:
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, window='hamming')
        Zxx = Zxx[:65, :]  # Ensure the frequency dimension matches the expected size (65)
        stft_data.append(np.abs(Zxx))
    return np.array(stft_data)

# Load data
base_path = os.getcwd()  # Update with your dataset path
data, labels = load_data(base_path)

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Apply STFT to normalized data
stft_data = apply_stft(normalized_data)

# Prepare input shapes
input_shape_time = (normalized_data.shape[1], 1)
input_shape_tf = (stft_data.shape[1], stft_data.shape[2], 1)

# Split data into training and validation sets
X_train_time, X_val_time, y_train, y_val = train_test_split(
    normalized_data[:, :, np.newaxis],  # Add a channel dimension for Conv1D input
    labels,
    test_size=0.2,
    random_state=42
)

X_train_tf, X_val_tf, _, _ = train_test_split(
    stft_data[:, :, :, np.newaxis],     # Add a channel dimension for Conv2D input
    labels,
    test_size=0.2,
    random_state=42
)

# Define the model architecture
input_time = Input(shape=input_shape_time)
input_tf = Input(shape=input_shape_tf)

# Time domain branch (1D CNN)
x_time = Conv1D(3, 16, activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_time)
x_time = ReLU()(x_time)
x_time = BatchNormalization()(x_time)
x_time = MaxPooling1D(2)(x_time)

x_time = Conv1D(3, 32, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x_time)
x_time = ReLU()(x_time)
x_time = BatchNormalization()(x_time)
x_time = MaxPooling1D(2)(x_time)

x_time = Conv1D(3, 64, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x_time)
x_time = ReLU()(x_time)
x_time = BatchNormalization()(x_time)

# Time-frequency domain branch (2D CNN for TF representation)
x_tf = Conv2D(3, (1, 16), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_tf)
x_tf = ReLU()(x_tf)
x_tf = BatchNormalization()(x_tf)
x_tf = MaxPooling2D((2, 1))(x_tf)

x_tf = Conv2D(3, (1, 32), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x_tf)
x_tf = ReLU()(x_tf)
x_tf = BatchNormalization()(x_tf)
x_tf = MaxPooling2D((2, 1))(x_tf)

x_tf = Conv2D(3, (1, 64), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x_tf)
x_tf = ReLU()(x_tf)
x_tf = BatchNormalization()(x_tf)

# Reshape TF output to match time domain for concatenation
x_tf = Reshape((x_tf.shape[1] * x_tf.shape[2], x_tf.shape[3]))(x_tf)

# Concatenate time and time-frequency branches
concat = Concatenate(axis=1)([x_time, x_tf])

# Bidirectional LSTM layers
blstm1 = Bidirectional(LSTM(20, return_sequences=True))(concat)
blstm2 = Bidirectional(LSTM(20, return_sequences=True))(blstm1)

# Flatten layer
flat = Flatten()(blstm2)

# Dropout layer
drop = Dropout(0.3)(flat)

# Dense layers
dense1 = Dense(128, activation='relu')(drop)
output = Dense(1, activation='sigmoid')(dense1)

# Create the model
model = Model(inputs=[input_time, input_tf], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    [X_train_time, X_train_tf], y_train,
    validation_data=([X_val_time, X_val_tf], y_val),
    epochs=100, batch_size=32, verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate([X_val_time, X_val_tf], y_val, verbose=0)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Print average accuracy
avg_train_accuracy = np.mean(history.history['accuracy'])
avg_val_accuracy = np.mean(history.history['val_accuracy'])
print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

# Calculate predictions
y_pred = model.predict([X_val_time, X_val_tf])
y_pred_classes = (y_pred > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_val, y_pred_classes, average='weighted')

# Calculate confusion matrix
conf_mat = confusion_matrix(y_val, y_pred_classes)
print(f"\nConfusion Matrix:\n{conf_mat}")

# Extract TN, FP, FN, TP from confusion matrix
tn, fp, fn, tp = conf_mat.ravel()

# Calculate Sensitivity
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# Calculate Specificity
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

# Calculate AUC
auc = roc_auc_score(y_val, y_pred)
print(f"AUC: {auc:.4f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training vs validation losses
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Display calculated metrics
print(f"Mean Train Accuracy: {avg_train_accuracy:.4f}")
print(f"Mean Val Accuracy: {avg_val_accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
