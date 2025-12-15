# Assignment 2 – Climate Forecasting

**Student Project @ TU Wien**  
**191.021 Introduction to Computational Sustainability, 2025 Winter Semester**

**Students:**  
Patrick Ennemoser · Dragana Sunaric · Daniel Martin Pühringer  

---

## Project Introduction

Climate change is increasing the frequency and severity of extreme weather events, making accurate and efficient climate forecasting essential for climate adaptation, risk mitigation, and infrastructure planning. Precipitation forecasting is particularly relevant due to its strong impact on hydrology, agriculture, flood management, and water resources.  

In this project, we investigate the effectiveness of different machine learning models for **next-day precipitation forecasting in Central Europe**. Using the **LamaH (Large-Sample Data for Hydrology and Environmental Sciences)** dataset, which provides over 35 years of meteorological time series and rich environmental attributes across 859 catchments, we compare classical statistical methods with neural network–based approaches. The goal is to analyze trade-offs between predictive accuracy, model complexity, and computational efficiency in the context of sustainable climate modeling.

---

## Evaluation

The evaluation is organized across multiple Jupyter notebooks, each focusing on a specific model or preprocessing step:

- **`preprocessing.ipynb`**  
  - Data loading and cleaning  
  - Selection of 100 random catchments  
  - Feature aggregation (daily statistics)
  - Naive Baseline forecast (MA & t-1)
  - Train/validation/test splitting  

- **`RidgeRegression.ipynb`**  
  - Baseline linear model with L2 regularization  
  - Serves as a low-complexity, energy-efficient reference model  

- **`Feed_Forward_NN.ipynb`**  
  - Multilayer Perceptron (FNN)  
  - Evaluation of non-linear relationships without temporal memory  

- **`LSTM.ipynb`**  
  - Recurrent Neural Network with Long Short-Term Memory  
  - Captures sequential dependencies in meteorological time series  
  - Multiple architectures and epoch settings evaluated  

- **`WaveNet.ipynb`**  *- best performance of all models*
  - Convolutional neural network with causal and dilated convolutions  
  - Designed to model long-range temporal dependencies efficiently  
  - Includes model finetuning and architectural optimization  

- **`plot_model_performance.ipynb`**  
  - Centralized comparison of all models  
  - Visualization of error metrics (e.g., RMSE)  
  - Aggregated performance plots used for final evaluation  

---

## Results

*TODO* 

### Model Performance Comparison

![Model Performance Comparison](model_performance.png)

*The plot compares the Root Mean Square Error (RMSE) of Ridge Regression, Feedforward Neural Network, LSTM, WaveNet and the fine-tuned WaveNet models. WaveNet achieves the lowest error, demonstrating superior performance on next-day precipitation prediction.*

---

## Repository Structure (Overview)

```text
.
├── data/
│   ├── preprocessing.ipynb
│   ├── RidgeRegression.ipynb
│   ├── Feed_Forward_NN.ipynb
│   ├── LSTM.ipynb
│   └── WaveNet.ipynb
├── plot_model_performance.ipynb
└── README.md
