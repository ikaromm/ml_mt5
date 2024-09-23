# ml_mt5

# Automated Trading Bot with Python and MetaTrader 5

![Project Banner](https://github.com/yourusername/automated-trading-bot/blob/main/banner.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Trading Bot](#running-the-trading-bot)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Contact](#contact)

## Overview

This project is an **Automated Trading Bot** developed in Python that integrates with **MetaTrader 5 (MT5)** to perform live trading operations based on machine learning predictions. The bot utilizes technical indicators and a **Random Forest** classifier to predict buy or sell signals for the **BTCUSDT** cryptocurrency pair on a 15-minute timeframe. This project serves as a personal learning tool to understand the integration of machine learning with trading platforms.

## Features

- **Data Acquisition**: Fetches historical candle data from MetaTrader 5.
- **Technical Indicators**: Calculates CCI, RSI, Bollinger Bands, MACD, and ATR.
- **Machine Learning**: Implements a Random Forest classifier with hyperparameter tuning using GridSearchCV.
- **Handling Imbalanced Data**: Uses SMOTE for oversampling to address class imbalance.
- **Model Evaluation**: Provides comprehensive evaluation metrics including accuracy, classification report, confusion matrix, AUC-ROC, and precision-recall curve.
- **Live Trading**: Utilizes the trained model to make real-time predictions and execute buy/sell orders on MT5.
- **Model Persistence**: Saves and loads the trained model using pickle for deployment.

## Technologies Used

- **Python**: Programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Matplotlib**: Data visualization.
- **MetaTrader5 (MT5)**: Trading platform API.
- **scikit-learn**: Machine learning library.
- **imbalanced-learn**: Handling imbalanced datasets with SMOTE.
- **pickle**: Model serialization.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7 or higher**: [Download Python](https://www.python.org/downloads/)
- **MetaTrader 5**: [Download MetaTrader 5](https://www.metatrader5.com/en/download)
- **MT5 Account**: An active account with MetaTrader 5.
- **Git**: For cloning the repository. [Download Git](https://git-scm.com/downloads)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/automated-trading-bot.git
cd automated-trading-bot
```

### 2. Create a Virtual Environment (Optional but Recommended)

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate

```

macOS and Linux

```
source venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure MetaTrader 5

Ensure that MetaTrader 5 is installed and properly configured on your machine. You may need to enable API access and obtain necessary credentials for connecting via the MetaTrader5 Python package.

## Usage

### Training the Model

1. **Configure MetaTrader 5 Connection** : Ensure MT5 is installed and running. Update the `symbol` and `timeframe` in the script if needed.
2. **Run the Training Script** : Execute the Python script to fetch data, calculate indicators, train the Random Forest model, and evaluate its performance.

```python
python train_model.py
```

1. *This will perform the following:*

   * Connect to MT5 and download historical candle data.
   * Calculate technical indicators (CCI, RSI, Bollinger Bands, MACD, ATR).
   * Prepare features and labels for the machine learning model.
   * Perform hyperparameter tuning using GridSearchCV with TimeSeriesSplit cross-validation.
   * Apply SMOTE for balancing the dataset.
   * Evaluate the model and visualize feature importances.
   * Save the trained model as `modelo_rf.pkl`.

### Running the Trading Bot

1. **Ensure the Trained Model Exists** : Make sure `modelo_rf.pkl` is present in the specified directory.
2. **Configure the Live Trading Script** : Update the path to the saved model in the trading script if necessary.
3. **Run the Trading Script** : Execute the Python script to start the live trading bot.

```
python trading_bot.py

```

   *This will perform the following:*

* Connect to MT5 and fetch the latest market data.
* Calculate the necessary technical indicators.
* Load the trained Random Forest model.
* Make predictions (buy/sell) based on live data.
* Execute buy or sell orders on MT5 according to the model's signal.

## Project Structure

```
automated-trading-bot/
│
├── train_model.py        # Script to train and evaluate the Random Forest model
├── trading_bot.py        # Script to run the live trading bot
├── modelo_rf.pkl         # Trained Random Forest model (generated after training)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # Project license

```



Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.
