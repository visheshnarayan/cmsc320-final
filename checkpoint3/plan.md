# Checkpoint 3: Model Development and Analysis

## 1. Introduction
We aim to classify non-verbal vocalizations from autistic individuals into expressive intent categories, using both traditional machine learning and deep learning approaches.

## 2. Machine Learning Models

### 2.1 Classical ML on Extracted Features
- **Logistic Regression**: Baseline model using Pitch Variance and MFCC-1/2/3.
- **Random Forest Classifier**: For capturing non-linearities and feature importance analysis.
- **XGBoost**: Boosted trees for faster and potentially better performance.
- **SVM (RBF Kernel)**: To separate non-linearly separable feature distributions.

*Features used:*
- Pitch Variance
- MFCC-1
- MFCC-2
- MFCC-3
- (Optional) Spectral Entropy

### 2.2 Deep Learning on Spectrograms
- **Simple CNN**:
  - 2–3 Conv2D layers + Pooling + Dense layers.
  - Treat spectrograms as 2D input images.

- **Attention-Augmented CNN**:
  - Add Squeeze-and-Excitation (SE) blocks or lightweight Self-Attention after convolution layers to focus on salient regions of the spectrograms.

### 2.3 Sequence Models
- **1D CNN** on cleaned audio waveforms directly.
- **Spectrogram → LSTM**:
  - Treat spectrogram frames as sequences.
  - Add an Attention layer over LSTM outputs to allow the model to focus on important time frames.

## 3. Advanced Ideas
- **Self-Supervised Pretraining**:
  - Pretrain embeddings via contrastive learning on spectrograms.
  - Train a simple classifier on these embeddings.

- **Clustering Analysis**:
  - Apply K-Means / DBSCAN on UMAP or t-SNE reduced feature spaces.
  - Visualize and validate natural groupings among vocalization labels.

## 4. Visualization Plan
- **Feature Importance**: For Random Forest/XGBoost models.
- **Confusion Matrix**: For each classification model.
- **Grad-CAM** or **Attention Maps**: For CNN models on spectrograms.
- **Embedding Plots**: UMAP/t-SNE scatter plots colored by label.

## 5. Insights and Conclusions
- Discuss:
  - Which features were most predictive?
  - Did deep learning significantly outperform classical ML?
  - How did attention mechanisms help focus learning?
  - Challenges faced (e.g., label imbalance, noisy samples).
  - Future ideas (e.g., data augmentation, larger models).

Input (Spectrogram) ➔
Conv2D ➔
BatchNorm ➔
ReLU ➔
Conv2D ➔
BatchNorm ➔
ReLU ➔
➔
[Attention Block across Time Axis]
➔
Global Average Pooling ➔
Dense ➔
Output
