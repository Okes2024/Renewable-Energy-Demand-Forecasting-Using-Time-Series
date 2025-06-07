# Renewable Energy Demand Forecasting Using Time Series

This project provides a solution for forecasting renewable energy demand using time series models, particularly Long Short-Term Memory (LSTM) networks.

## Overview
- **Task**: Time Series Forecasting of renewable energy demand.
- **Model**: LSTM (Long Short-Term Memory) Neural Network.
- **Data**: Historical energy demand data.
- **Libraries**: TensorFlow, Keras, Pandas, Matplotlib, scikit-learn.

## Project Structure
- `data.csv`: CSV file containing historical energy demand data.
- `prepare_data()`: Function to prepare the time series dataset.
- `build_lstm_model()`: Function defining the LSTM model.
- `train_test_split()`: Split data into training and testing sets.
- `train_model()`: Function to train the model.
- `forecast()`: Function to make future predictions.

## How to Run
1. Prepare a dataset (`data.csv`) with timestamps and demand values.
2. Load and preprocess the dataset.
3. Train the LSTM model.
4. Evaluate and forecast future energy demand.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Pandas
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install tensorflow pandas matplotlib scikit-learn
```

## Output
- Trained LSTM model for demand forecasting.
- Forecasted energy demand plot.

## Author
**Okes Imoni**

---
Feel free to fork and contribute to the project!
