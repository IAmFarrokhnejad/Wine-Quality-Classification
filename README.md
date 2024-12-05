# Wine-Quality-Classification
 Wine Quality Classification Using Deep Learning


This repository implements a comparative study of three deep learning architectures for wine quality classification: **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory Networks (LSTMs)**, and **Autoencoders**. Using physicochemical attributes from red and white wine datasets, the models are evaluated based on their performance metrics such as precision, recall, F1-score, and accuracy.

## Datasets

The models utilize the Wine Quality datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). The datasets, `winequality-red.csv` and `winequality-white.csv`, must be placed in the `Data/` directory for the scripts to work.

## Features

- **CNN:** Leverages convolutional layers to capture spatial patterns in the data.
- **LSTM:** Explores temporal relationships in the physicochemical attributes.
- **Autoencoder:** Utilizes dimensionality reduction for feature extraction and classification.

## Requirements

- Python 3.7+
- Libraries: `pandas`, `numpy`, `torch`, `sklearn`, `seaborn`, `matplotlib`, `imblearn`

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Scripts
1. CNN.py: Implements a CNN for wine quality classification.
2. LSTM.py: Implements an LSTM for classification tasks.
3. Autoencoders.py: Uses an Autoencoder for feature extraction and a secondary classifier for prediction.

### Each script includes:

- Data preprocessing, including class balancing with SMOTE.
- K-fold cross-validation for robust performance evaluation.
- Model training and evaluation.


### Usage 
1. Make sure the dataset paths are valid (First section of the code).
2.  Run any of the scripts:

```bash
python 1_CNN.py
python 2_LSTM.py
python 3_Autoencoders.py
```

## Observations
CNNs outperformed other models, achieving the highest accuracy, precision, recall, and F1-scores.
LSTMs demonstrated competitive performance, showcasing their utility in classification tasks.
Autoencoders struggled due to feature compression, leading to a loss of variance between classes.

## Conclusion
This study highlights the effectiveness of deep learning models for wine quality classification:

CNNs are most suitable for structured tabular data.
LSTMs offer adaptability and robust results.
Autoencoders, while useful for dimensionality reduction, may compromise classification accuracy.
Future work could address challenges like class imbalance and limited dataset size by exploring advanced oversampling techniques and data augmentation.

####
Developed by Morteza Farrokhnejad.