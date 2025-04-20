# Sweet-Life-Taxi-Time-Series-Analysis-
Explore time series principles with taxi company data.  

🚕 Sweet Life Taxi Data – Time Series Analysis

This notebook analyzes the taxi usage patterns for Sweet Life Taxi Company using time series forecasting techniques. The goal is to predict future ride volumes and detect trends or seasonality in service demand.

📚 Table of Contents
About the Project

Installation

Usage

Project Structure

Technologies Used

Results & Insights

Screenshots

Contributing

License

📌 About the Project
The notebook walks through:

Loading and cleaning historical ride data

Visualizing trends and seasonality

Applying rolling averages and differencing

Building a forecasting model using classical and/or statistical approaches (e.g. SARIMA)

Evaluating model performance and making recommendations

🛠 Installation
Clone or download this repository

Install required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn statsmodels jupyter
Run the notebook:

bash
Copy
Edit
jupyter notebook

🚀 Usage
Open Sweet Life Taxi Data (Time Series Analysis).ipynb in Jupyter and run each cell. The notebook includes:

Time series decomposition

Trend and seasonality isolation

Forecasting future demand

Visualization and model evaluation

📁 Project Structure
bash
Copy
Edit
Sweet Life Taxi Data (Time Series Analysis).ipynb  # Main analysis notebook
README.md                                         # Documentation
images_sweetlife/                                 # Screenshots folder

⚙️ Technologies Used
Python 3.8+

Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Statsmodels

📊 Results & Insights

Taxi usage peaks on weekends and evenings

Clear weekly seasonality pattern detected

Final forecast model achieved a Mean Absolute Error (MAE) of ~15.3 rides/day

GridSearchCV was able to provide the best hyperparameters for the different types of models being utilized.  Training set evaluated Random Forest, Decision Tree, Gradient Boosting, and LightGBM.  In analyzing performance on the test set the best model was Gradient Boosting with an RMSE of 27.54. Grdient Boosting was then run with the test set data.  The final RMSE was 27.54.

📸 Screenshots


🤝 Contributing
Ideas for improving this project? Want to add Prophet, LSTM, or hybrid models? Fork the repo and submit a pull request!

🪪 License
This project is licensed under the MIT License.  


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
