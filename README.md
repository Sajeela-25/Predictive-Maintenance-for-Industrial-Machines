# Predictive Maintenance for Industrial Machines

A machine learning-based predictive maintenance system that predicts whether an industrial machine is likely to fail based on sensor readings.

This project uses the **AI4I 2020 Predictive Maintenance Dataset** and a **Random Forest Classifier** for binary machine-failure prediction. It also includes an interactive **Streamlit dashboard** for uploading sensor data, generating predictions, and viewing visualizations.

---

## Project Overview

Unexpected machine failures can stop production, waste materials, delay deliveries, and increase maintenance costs.

The purpose of this project is to use machine sensor data to identify machines that may be at risk of failure.

The system performs:

- Data preprocessing
- Missing-value handling
- Binning / discretization
- Outlier handling using IQR
- Feature scaling
- Feature selection
- PCA dimensionality reduction
- Random Forest classification
- Model evaluation
- Data visualization
- Interactive Streamlit interface

---

## Objectives

The main objectives of this project are:

1. Analyze industrial machine sensor data.
2. Prepare the data for machine learning.
3. Identify important sensor features.
4. Detect and handle unusual values.
5. Train a machine learning model to predict failures.
6. Evaluate the model using different performance metrics.
7. Provide an easy-to-use interface for predictions and visualization.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains approximately **10,000 records** of industrial machine measurements.

### Features Used

| Feature | Description |
|---|---|
| Air Temperature | Temperature around the machine |
| Process Temperature | Temperature during the machine process |
| Rotational Speed | Machine rotational speed in RPM |
| Torque | Torque applied by the machine |
| Tool Wear | Amount of tool wear in minutes |
| Machine Failure | Target variable |

### Target Variable

```text
0 = No Failure
1 = Failure
