# Machine Learning the Unspoken  
**Acoustic Feature Analysis for Classifying Intent in Nonverbal Vocalizations**  

### Authors  
- Vishesh Narayan: [`git@visheshnarayan`](https://github.com/visheshnarayan)  
- Shivam Amin: [`git@ShivAmamin05`](https://github.com/ShivAmamin05)  
- Deval Bansal[`git@devnban`](https://github.com/devnban)  
- Eric Yao[`git@eyao6`](https://github.com/eyao6)  
- Eshan Khan [`git@eshan327`](https://github.com/eshan327)

---

## Project Summary

This project explores how machine learning can decode expressive intent from nonverbal vocalizations in autistic individuals. We use audio from the [ReCANVo dataset](https://doi.org/10.5281/zenodo.5786860), applying a comprehensive pipeline including denoising, normalization, silence removal, feature extraction, spectrogram generation, and supervised classification.

---

## Dataset

**Dataset**: [ReCANVo](https://doi.org/10.5281/zenodo.5786860)  
Contains real-world audio samples labeled with categories like “yes”, “no”, “frustrated”, “delighted”, and others.  

**Source**: Johnson, K. T., Narain, J., Quatieri, T., et al. (2023). *ReCANVo: A Database of Real-World Communicative and Affective Nonverbal Vocalizations*. Sci Data 10, 523. [DOI](https://doi.org/10.1038/s41597-023-02405-7)

---

## Preprocessing Pipeline

We designed a modular preprocessing pipeline that includes:
- Audio cleaning (denoising, silence trimming, normalization)
- Feature extraction (pitch, MFCCs, entropy)
- Spectrogram generation (Mel-scale, dB conversion)
- Padding to fixed-size matrices
- Global stats computation (z-score or min-max normalization)

The entire pipeline is implemented using `librosa`, `polars`, and `numpy` with multithreaded loading for speed.

---

## Exploratory Data Analysis

- Analyzed distribution of labels across over 7,000 samples  
- Examined participant-level label diversity  
- Computed statistics on audio lengths  
- Visualized Mel spectrograms for each label  
- Found large inter-participant variation in audio duration and label type

---

## Modeling Approaches

We evaluated several model families:

### Classical Models
- K-Means Clustering
- Hierarchical Ensemble of SVMs (One-vs-One)

### Deep Learning
- Convolutional Neural Networks (CNNs)
- Vision Transformers (ViTs)

ViTs were particularly effective due to their ability to learn global features across padded spectrograms.

---

## Results

- Accuracy increased with the use of synthetic data and class-weighted loss
- Vision Transformers showed high robustness to sparse or padded input
- CNNs performed well but struggled with low-resource labels
- SVMs offered interpretability but lower generalization

---

## Project Assets

- [Final Report Website](https://visheshnarayan.github.io/)

---

## References

- van der Maaten, L., & Hinton, G. (2008). [Visualizing Data Using t-SNE](http://jmlr.org/papers/v9/vandermaaten08a.html). *Journal of Machine Learning Research*, 9(86), 2579–2605.
- McInnes, L., Healy, J., & Melville, J. (2020). [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426). arXiv.
- Dosovitskiy, A., et al. (2021). [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929). arXiv.
- Palanisamy, K., Singhania, D., & Yao, A. (2020). [Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154). arXiv.
- Robinson, D., Miron, M., Hagiwara, M., & Pietquin, O. (2024). [NatureLM-audio](https://arxiv.org/abs/2411.07186). arXiv.
- Brownlee, J. (2021). [One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/). MachineLearningMastery.com.
