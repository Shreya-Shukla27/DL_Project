# 🌞 Solar Power Generation Prediction using LSTM

A deep learning project that predicts solar power output (in kW) from meteorological features using a stacked LSTM neural network built with TensorFlow/Keras.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)

---

## Overview

This project uses historical weather and solar data to train an LSTM model capable of forecasting solar power generation. The model takes a sequence of 24 previous time steps as input and predicts the power output for the next step- useful for grid management, energy storage planning, and solar farm operations.

---

## Dataset

- **File:** `solarpowergeneration.csv`
- **Rows:** 4,213 | **Columns:** 21
- **Target:** `generated_power_kw`
- **No missing values**

### Selected Features

| Feature | Description |
|---|---|
| `temperature_2_m_above_gnd` | Air temperature (°C) |
| `relative_humidity_2_m_above_gnd` | Relative humidity (%) |
| `mean_sea_level_pressure_MSL` | Atmospheric pressure (hPa) |
| `total_precipitation_sfc` | Precipitation (mm) |
| `total_cloud_cover_sfc` | Cloud coverage (%) |
| `shortwave_radiation_backwards_sfc` | Solar radiation (W/m²) |
| `wind_speed_10_m_above_gnd` | Wind speed (m/s) |
| `angle_of_incidence` | Sun-panel angle (°) |
| `zenith` | Solar zenith angle (°) |
| `azimuth` | Solar azimuth angle (°) |

---

## Project Structure

```
solar-power-lstm/
│
├── solarpowergeneration.csv        # Dataset
├── solar_power_lstm.ipynb          # Main notebook
├── README.md
└── requirements.txt
```

---

## Model Architecture

A two-layer stacked LSTM with regularization to prevent overfitting:

```
Input (24 timesteps × 10 features)
    ↓
LSTM (50 units, return_sequences=True) + L2 regularization
    ↓
BatchNormalization → Dropout (0.5)
    ↓
LSTM (50 units) + L2 regularization
    ↓
Dropout (0.5)
    ↓
Dense (1) → Predicted Power (kW)
```

**Total parameters:** 32,651

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | MSE |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| LR reduction factor | 0.5 (patience=5) |
| Train / Val / Test split | 64% / 16% / 20% |

---

## Results

| Metric | Value |
|---|---|
| R² Score | 0.5524 |
| RMSE | 608.18 kW |
| MAE | 475.98 kW |
| Explained Variance | 0.6039 |

The model converged at epoch 100 with a final validation loss of **0.2892** and validation MAE of **0.3437** (on scaled data).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/solar-power-lstm.git
cd solar-power-lstm

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.20.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Usage

### Run the notebook
Open `solar_power_lstm.ipynb` in Jupyter and run all cells.

### Interactive Prediction
At the end of the notebook, a simple CLI predictor lets you enter live weather conditions:

```
☀️  Solar Radiation (W/m²): 800
🌡️  Temperature (°C): 25
☁️  Cloud Cover (%): 20
💨 Wind Speed (m/s): 3.5
💧 Humidity (%): 60

⚡ PREDICTED SOLAR POWER: 2389.12 kW
🌟 Excellent! Great conditions for solar power generation!
```

---

## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**- LSTM model
- **scikit-learn**- preprocessing, metrics
- **Pandas / NumPy**- data handling
- **Matplotlib / Seaborn**- visualization

---

## 📌 Notes

- Data is scaled using `StandardScaler` for both features and target
- Sequences of length 24 are used to capture temporal dependencies
- MAPE is inflated due to near-zero power values at night/low-light hours; RMSE and R² are more reliable indicators of model quality
