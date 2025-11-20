# Bulldozer Price Prediction
This project aims to predict the auction sale price of heavy equipment (e.g., bulldozers) based on historical data. The notebook walks through the end-to-end workflow of loading the data, performing exploratory data analysis (EDA), building machine learning models, and evaluating their performance using key regression metrics.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Dependencies](#dependencies)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Preprocessing](#preprocessing)
7. [Modeling](#modeling)
8. [Evaluation](#evaluation)
9. [Scikit-Learn Workflow](#scikit-learn-workflow)

## Project Overview
The objective of this project is to predict the auction sale price of bulldozers and other heavy machinery at auction using machine learning algorithms. The following approach is used:

1. **Problem Definition**: 
   - Can we predict the future sale price of a bulldozer given its features, historical data, and auction information? This problem is a regression task since we aim to predict continuous values (prices).

2. **Data**: 
   - The dataset is sourced from the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data).
   - The data consists of three main datasets:
      - Train.csv: Historical data up to the end of 2011, used for training the model.
      - Valid.csv: Data from January to April 2012, used for validation and leaderboard scoring.
      - Test.csv: Data from May to November 2012, used for final competition rankings.
   - These datasets contain numerous features such as product details, machine specifications, usage metrics, and the sale date.

3. **Evaluation**: 
   - The performance metric is **Root Mean Squared Log Error (RMSLE)**, which measures the log-scaled error between predicted and actual prices.
   - The model achieved an **RMSLE of 0.2467**, ranking **32nd out of 474 participants** in the competition.

## Dataset
The dataset contains the following features:

- **YearMade**: The year the machine was manufactured.
- **MachineHoursCurrentMeter**: The number of hours the machine has been used.
- **UsageBand**: The usage range (Low, Medium, High).
- **fiModelDesc**: Full description of the machine model.
- **fiBaseModel**: Base model of the machine.
- **fiProductClassDesc**: Product class of the equipment.
- **DriveSystem**: The drive system used (e.g., 4WD).
- **Enclosure**: The type of enclosure (e.g., cab or canopy).
- **Hydraulics**: Whether the machine has hydraulics.
- **Tire_Size**: The tire size of the machine.
- **SaleDate**: The date of the auction sale.
- **SalePrice**: The auction sale price (target variable).
- And many more.

## Installation
To run this project, you will need to install the following libraries:

```bash
# Clone the repository
git clone https://github.com/yourusername/bulldozer-price-prediction.git

# Navigate to the project directory
cd bulldozer-price-prediction

# Install dependencies
pip install -r requirements.txt
```

## Dependencies:
- Pandas
- Numpy
- Matplotlib

## Exploratory Data Analysis (EDA)
In the EDA phase, the notebook focuses on:

- Handling missing values and outliers.
- Visualizing feature distributions and identifying relationships between variables.
- Investigating machine attributes (e.g., age, usage) and their correlation with the target variable (`SalePrice`).
- Feature extraction and engineering to optimize model performance (e.g., breaking down `SaleDate` into year, month).

## Preprocessing
- **Handling Missing Data**: Missing values are addressed using appropriate imputation techniques (e.g., filling with mean/median).
- **Feature Engineering**: Categorical variables are encoded, and time-related features are extracted from the `SaleDate` column.
- **DateTime Features**: The `SaleDate` is split into separate year, month, and day features to better capture time-based trends in auction prices.

## Modeling
The following machine learning algorithms are used:
- **Random Forest Regressor**: A robust ensemble method known for its ability to handle large datasets and complex relationships between variables.
   - **Hyperparameter Tuning**:
      - Both **RandomizedSearchCV** and **GridSearchCV** were used to find the optimal configuration for the Random Forest model.
   - **Training and Validation**:
      - The model was trained on `Train.csv` and validated on `Valid.csv`. The focus was on optimizing model performance through careful feature selection and tuning.

## Evaluation
- **Metric**: The model’s performance is evaluated using **RMSLE (Root Mean Squared Log Error)**.
- The final model achieved an **RMSLE of 0.2467** on the validation set, which places it in the **top 7% of participants** (32nd out of 474) in the Kaggle competition leaderboard.

## Scikit-Learn Workflow
1. Get data ready
2. Pick a model(to suit your problem)
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model
