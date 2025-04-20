# Predicting Disordered Regions in Proteins

This repository contains the code and methodologies developed during the CBIO Hackathon 2025 for predicting intrinsically disordered regions (IDRs) in proteins using sequence-based features and structural insights.

## Contributors
- **Noa Birman:** Data collection, HMM model, and pLDDT
- **Dana Siton:** Data preprocessing and analyses
- **Tsori Kislev:** ESM2 embeddings and Logistic Regression model
- **Tal Neumann:** ESM2 embeddings and Multi-Layer Perceptron (MLP) model

## Project Overview

Intrinsically disordered regions (IDRs) play crucial roles in biological processes like molecular recognition, signaling, and regulation. This project leverages computational methods to predict these regions using sequence data and structural predictions.

## Data

The dataset used is from the DisProt database (version 9.7), providing annotations for intrinsically disordered proteins:
- Proteins: 3,312 initially, reduced to 1,360 after preprocessing
- Mean sequence length: 435.5 amino acids
- Mean disorder percentage: 40%

### Preprocessing Steps
- Filtering based on AlphaFold structural predictions
- Selecting sequences between 100-1000 amino acids
- Selecting proteins with at least one disordered region of ≥30 amino acids

## Models

### 1. Hidden Markov Model (HMM)
- 2-state model (Ordered vs. Disordered)
- Achieved Balanced Accuracy: 0.73; AUC: 0.79

### 2. Logistic Regression
- Utilized features from ESM2 protein embeddings
- Achieved Balanced Accuracy: 0.74; AUC: 0.83

### 3. Multi-Layer Perceptron (MLP)
- Feedforward neural network using ESM2 embeddings
- Optimized hyperparameters via Optuna
- Achieved Balanced Accuracy: 0.79; AUC: 0.84

## AlphaFold Confidence Scores (pLDDT)
- Explored correlation between AlphaFold pLDDT scores and disorder
- Moderate correlation (Spearman correlation: -0.348)
- Found pLDDT scores helpful but insufficient alone as definitive predictors

## Model Performance Comparison

| Model                  | Balanced Accuracy | AUC  |
|------------------------|-------------------|------|
| **MLP (Ours)**         | **0.79**          | **0.84** |
| Logistic Regression    | 0.74              | 0.82 |
| HMM                    | 0.73              | 0.79 |
| AlphaFold Confidence   | 0.72              | 0.75 |
| IUPred                 | 0.78              | 0.76 |
| DISOPRED2              | 0.62              | 0.74 |
| DISOPRED3              | 0.79              | 0.90 |

## Next Steps
- Incorporate more structural data from AlphaFold
- Expand dataset with additional annotations
- Explore hybrid models (MLP + HMM or transformers)
- Improve interpretability with visualization tools
- Benchmark on independent datasets

## References

- Aspromonte et al., 2024 - DisProt in 2024: improving function annotation of intrinsically disordered proteins.
- Lin et al., 2022 - Evolutionary-scale prediction of atomic level protein structure with a language model.
- Jumper et al., 2021 - Highly accurate protein structure prediction with AlphaFold.
- Varadi et al., 2022 - AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models.
- Dosztányi et al., 2005 - IUPred: web server for the prediction of intrinsically unstructured regions of proteins.
- Jones & Cozzetto, 2015 - DISOPRED3: precise disordered region predictions with annotated protein-binding activity.

For more detailed references, see the full documentation 
[ProModel.pdf](https://github.com/user-attachments/files/19824792/ProModel.pdf)
