# SCDex: Construction Works Index & Slippage Forecaster

SCDex (Construction Works Index) is a Python-based web application and dashboard designed to track, visualize, and forecast construction project slippage. Built with **Dash** and **Plotly**, it leverages **Gaussian Process Regression (GPR)** to predict future project delays and provides a robust 95% Confidence Interval for its forecasts.

## 🚀 Features

* **Interactive Dashboard:** A responsive UI built with Dash and Dash Bootstrap Components, allowing users to filter data by Contractor and Project.
* **Advanced Forecasting:** Uses Scikit-Learn's `GaussianProcessRegressor` with custom kernels (WhiteKernel, ExpSineSquared, RBF) to predict time-series slippage for construction works.
* **Confidence Intervals:** Automatically calculates and visualizes 95% Confidence Bounds (Upper and Lower) around the predicted forecast using Plotly's `tonexty` shading.
* **Custom Walk-Forward Validation:** Employs a tailored time-series cross-validation method to ensure the machine learning model is trained and tested accurately on sequential project data.
* **Lightweight Architecture:** Designed to run without heavy database dependencies (PostgreSQL removed), utilizing local flat files (`.csv`) and in-memory Pandas dataframes for quick deployment.

## 🛠️ Tech Stack

* **Frontend:** [Dash](https://dash.plotly.com/), [Plotly Graph Objects](https://plotly.com/python/graph-objects/), Dash Bootstrap Components
* **Backend/Data Processing:** Python 3.x, [Pandas](https://pandas.pydata.org/), NumPy
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Gaussian Processes, MinMaxScaler, train_test_split)

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/cuburt/SCDex.git](https://github.com/cuburt/SCDex.git)
cd SCDex
```

2. **Create a virtual environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies:**
Ensure you have the required libraries installed. You can install them manually or via a `requirements.txt` file if generated:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn
```

4. **Add your dataset:**
Ensure your historical data is located in the correct directory. The app expects a CSV file at:
`dataset/full_slippage_dataset.csv`

## 💻 Usage

To launch the application, run the main Python script (usually `app.py` or `index.py` depending on your entry point):

```bash
python main.py
```

* Open your web browser and navigate to `http://127.0.0.1:8050/`.
* Use the dropdowns to select a **Contractor** and a specific **Project**.
* The dashboard will automatically upsample the data, train the GPR model, and render the historical slippage alongside the shaded forecast.

## 📊 Data Preprocessing Details

The application performs several automated data cleaning steps tailored for construction indices:

* Cleans numeric columns (stripping commas and standardizing types for contract amounts and days).
* Upsamples historical data to a daily frequency using Spline interpolation.
* Employs Forward Fill (`ffill`) to handle categorical data and trailing missing values.
* Scales inputs using `MinMaxScaler` before feeding them into the Gaussian Process model.

