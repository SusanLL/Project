# Project

## Overview
This repository contains a collection of data-science notebooks demonstrating various machine-learning techniques. Each notebook focuses on a distinct dataset and modeling challenge, offering self-contained examples suitable for review by HR, recruiters, or technical stakeholders.

## Contents
| Notebook | Description | Key Libraries |
|----------|-------------|---------------|
| **Brain_MRI_Images_for_Brain_Tumor_Detection.ipynb** | Trains a simple Convolutional Neural Network (CNN) to classify brain MRI images as tumorous or non-tumorous. | `tensorflow`, `keras`, `numpy`, `sklearn`, `matplotlib`, `datasets` |
| **Breast_Cancer_Wisconsin.ipynb** | Exploratory analysis and multiple classification models (e.g., decision trees, random forest, XGBoost) for the UCI Breast Cancer Wisconsin dataset. | `pandas`, `numpy`, `matplotlib`, `seaborn`, `ucimlrepo`, `sklearn`, `xgboost`, `statsmodels` |
| **Estimation_of_Obesity_Levels_Based_On_Eating_Habits_and_Physical_Condition.ipynb** | Predicts obesity levels using demographic and lifestyle features from a UCI dataset; includes feature exploration and model training. | `pandas`, `seaborn`, `matplotlib`, `sklearn`, `scipy`, `ucimlrepo` |
| **Recommendation_System_Assignment.ipynb** | Builds an item-based collaborative filtering recommender system using the MovieLens 100k dataset with the Surprise library. | `numpy`, `pandas`, `surprise` |

## Getting Started
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project
   ```
2. **Set up a Python environment** (Python 3.8+ recommended) and install required packages. Example using `pip`:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn xgboost surprise
   ```
3. **Launch Jupyter Notebook or open in Google Colab**
   ```bash
   jupyter notebook
   ```
   Then open any of the `.ipynb` files to explore the analyses.

## Dataset Sources
- **Brain MRI Images for Brain Tumor Detection** – public datasets of MRI scans.
- **Breast Cancer Wisconsin (Diagnostic)** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **MovieLens 100k** – [GroupLens](https://grouplens.org/datasets/movielens/100k/)

## Contributing
Contributions are welcome. Please fork the repository, make your changes in a new branch, and submit a pull request for review.

## License
No license file is provided. If you intend to use or distribute this work, please add an appropriate license.
