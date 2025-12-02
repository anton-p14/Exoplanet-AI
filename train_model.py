import pandas as pd

# Load the dataset
try:
	# This file includes many metadata/comment lines that start with '#'.
	# Use comment='#' so pandas ignores those lines and reads the real header row.
	# Use the python engine and a forgiving bad-lines policy to avoid C-parser tokenization errors
	# caused by irregular rows in some archive files.
	try:
		# Prefer the python engine for robust parsing of irregular archive rows
		df = pd.read_csv(
			'kepler_data.csv',
			comment='#',
			sep=',',
			engine='python',
			on_bad_lines='warn',
		)
	except TypeError:
		# Some pandas versions may not accept on_bad_lines with the python engine.
		# Fall back to the C engine and skip problematic lines.
		df = pd.read_csv(
			'kepler_data.csv',
			comment='#',
			sep=',',
			engine='c',
			error_bad_lines=False,
			warn_bad_lines=True,
		)
except Exception as e:
	print(f"Failed to read 'kepler_data.csv': {e}")
	raise

# Show the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Show all column names
print("\nColumn names:")
print(df.columns)

# Show basic info
print("\nData info:")
print(df.info())

# Step 1 ----------------------------------------

# Select useful features and target
features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_srad', 'koi_smass']
target = 'koi_disposition'

# Drop rows with missing values in selected columns
df_clean = df[features + [target]].dropna()

# Encode target labels (convert text to numbers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])



# Phase 2----------------------------------------------------------

def habitability_score(row):
    score = 1.0

    # Penalize large planets
    if row['koi_prad'] > 2: score *= 0.5
    elif row['koi_prad'] < 0.5: score *= 0.7

    # Penalize unstable stars
    if row['koi_srad'] > 2: score *= 0.7
    if row['koi_smass'] > 2: score *= 0.7

    # Penalize extreme orbital periods
    if row['koi_period'] < 10 or row['koi_period'] > 500: score *= 0.6

    return round(score, 2)

# Apply to cleaned dataset
df_clean['habitability_index'] = df_clean.apply(habitability_score, axis=1)

# Show sample output
print("\nSample Habitability Scores:")
print(df_clean[['koi_prad', 'koi_srad', 'koi_smass', 'koi_period', 'habitability_index']].head())
 
#----------------------------------------------------------




# Split into input (X) and output (y)
X = df_clean[features]
y = df_clean[target]



# Step 2 ----------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Phase 1--------------------------------------
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Plot and save
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Exoplanet Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature importance plot as 'feature_importance.png'")
#-------------------------------------------------

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")



# Phase 3 --------------------------------------------------

# Get prediction probabilities
y_proba = model.predict_proba(X_test)

# Set a confidence threshold
threshold = 0.7

# Flag low-confidence predictions
low_confidence_flags = [max(probs) < threshold for probs in y_proba]

# Create a results DataFrame
import numpy as np
results_df = pd.DataFrame({
    'Prediction': le.inverse_transform(y_pred),
    'Confidence': [round(max(probs), 2) for probs in y_proba],
    'LowConfidence': low_confidence_flags,
	# Use label-based .loc because X_test.index contains the original row labels, not integer positions.
	'HabitabilityIndex': np.round(df_clean.loc[X_test.index, 'habitability_index'].values, 2)
})

# Show sample output
print("\n🔭 Discovery Mode Preview:")
print(results_df.head())

#----------------------------------------------------------




# Step 3 ----------------------------------------

import joblib
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")
