# 🌌 Exoplanet Classification AI Model

A machine learning project that classifies exoplanet candidates using NASA's Kepler mission data, featuring a custom habitability scoring system.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project uses Random Forest classification to analyze exoplanet candidates from the Kepler Space Telescope dataset. It predicts whether a Kepler Object of Interest (KOI) is a confirmed exoplanet, false positive, or candidate, while also calculating a custom habitability index based on planetary and stellar characteristics.

## ✨ Features

- **Binary & Multi-class Classification**: Classifies exoplanet candidates using 6 key features
- **Habitability Scoring**: Custom algorithm evaluating planetary habitability based on:
  - Planetary radius
  - Stellar radius and mass
  - Orbital period
- **Confidence Tracking**: Flags low-confidence predictions for further review
- **Feature Importance Analysis**: Visual representation of which features matter most
- **Model Persistence**: Trained model saved for future predictions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/exoplanet-ai-model.git
cd exoplanet-ai-model
```

2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Or install all at once:
```bash
pip install pandas==2.3.2 numpy==2.3.3 scikit-learn==1.7.2 matplotlib==3.10.6 joblib==1.5.2
```

### Dataset

The project uses NASA's Kepler Exoplanet Archive data (`kepler_data.csv`). The dataset includes:
- **Source**: NASA Exoplanet Archive
- **Features Used**:
  - `koi_period`: Orbital period (days)
  - `koi_duration`: Transit duration (hours)
  - `koi_depth`: Transit depth (ppm)
  - `koi_prad`: Planetary radius (Earth radii)
  - `koi_srad`: Stellar radius (Solar radii)
  - `koi_smass`: Stellar mass (Solar masses)
- **Target**: `koi_disposition` (CONFIRMED, FALSE POSITIVE, CANDIDATE)

## 💻 Usage

Run the training script:

```bash
python train_model.py
```

The script will:
1. Load and clean the Kepler dataset
2. Engineer a habitability index feature
3. Train a Random Forest classifier
4. Generate feature importance visualization (`feature_importance.png`)
5. Display prediction results with confidence scores
6. Save the trained model (`model.pkl`)

### Sample Output

```
Model Accuracy: 0.99

🔭 Discovery Mode Preview:
     Prediction  Confidence  LowConfidence  HabitabilityIndex
0  FALSE POSITIVE    0.99        False              0.42
1     CONFIRMED       0.85        False              0.70
2     CANDIDATE       0.65        True               0.35
```

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Test Set Size**: 20% of data
- **Typical Accuracy**: ~99% on Kepler dataset

## 🧠 Habitability Index

The custom habitability scoring algorithm evaluates potential habitability based on:

```python
Base Score: 1.0

Penalties:
- Large planets (>2 Earth radii): ×0.5
- Small planets (<0.5 Earth radii): ×0.7
- Large stars (>2 Solar radii): ×0.7
- Massive stars (>2 Solar masses): ×0.7
- Extreme orbits (<10 or >500 days): ×0.6
```

Higher scores indicate better habitability potential (theoretical, simplified model).

## 📁 Project Structure

```
exoplanet-ai-model/
│
├── train_model.py              # Main training script
├── kepler_data.csv            # Kepler mission dataset
├── model.pkl                  # Trained model (generated)
├── feature_importance.png     # Feature visualization (generated)
├── TOI_2025.10.04_01.06.40.csv
├── k2pandc_2025.10.04_01.07.19.csv
└── README.md                  # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NASA Exoplanet Archive for providing the Kepler mission data
- The scikit-learn community for excellent ML tools
- Kepler Space Telescope team for the groundbreaking exoplanet discoveries

⭐ **Star this repo if you find it helpful!**
