# Predicting Movie Performance Through IMDb Rating Classification


## General Overview
**Goal:**
- Create a classification model capable of predicting viewer's rating on movies based on a ternary categorical variable (satisfied/not satisfied)
- Identify prominent factors from the model that affects movie performance ratings
  
**Data Source:**
- https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset
- Shape = 5,043 rows and 28 columns
- Variable Examples: Director names, movie budget, content rating (PG-13, R), actor facebook likes, IMDb score, number of user reviews

**Models Tested:**
- Decision Tree
- Random Forest

**Model Optimization Methodologies:**
- Synthetic Minority Oversampling Technique (SMOTE)
- Hyperparameter Tuning using GridSearch
- Cost Complexity Pruning
- Wrapper Method Feature Selection
- Binning Contineous Variables


## Table of Contents 
- [Data](#data)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Benchmarking and Optimization](#model-benchmarking-and-optimization)
- [Model Comparison and Interpretation](#model-comparison-and-interpretation)
- [Final Remarks](#final-remarks)


