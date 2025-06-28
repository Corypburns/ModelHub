import os, pandas as pd, time, matplotlib.pyplot as mpl, csv, time, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from array import array
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

#TODO: Still need to import the memory consumption code, since I do not own a Jetson, I have to do this in the lab; this will be the easy step
#TODO: Need to change the writings of the csv file to match what the other csv files are written as

# Paths
output_path = '/home/cory/code/Anik-Research-Testing/ModelHub/OUTPUTS/Text-Classification/RandomForest' 
dataset_path = '/home/cory/code/Anik-Research-Testing/DATASETS/Text-Based Datasets/Text-Classification/GAMBLERYU/GAMBLERYU.csv'
os.makedirs(output_path, exist_ok=True)

# Timestamp and output file
now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
output_file = os.path.join(output_path, f"RandomForest_{now}.csv")

# Load dataset
df = pd.read_csv(dataset_path)
df = df[['Query', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)

# Inputs and labels
input_data = df['Query']
label_data = df['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

# TF-IDF
vector = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_vec, y_train)

# Joblib to save model to get time
model_path = os.path.join(output_path, f"RandomForestModel_{now}.joblib")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Start model load
start = int(time.time() * 1000)
model = joblib.load(model_path)
end = int(time.time() * 1000)
print(f"Model loaded in {end - start} ms...")

# Predict
prediction = model.predict(X_test_vec)
conf = model.predict_proba(X_test_vec)
X_index = X_test.index

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
count = Counter(y_test)
pos = count[1]
neg = count[0]

# Write predictions
try:
    with open(output_file, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Prediction", "Confidence", "Actual"])
        for idx, (i, pred) in enumerate(zip(X_index, prediction)):
            query = df.loc[i, 'Query']
            label = df.loc[i, 'Label']
            confidence = conf[idx][pred] * 100
            print(f"\nQuery: {query}\nLabel: {label}\nPrediction: {pred}\nConfidence: {confidence:.2f}")
            writer.writerow([query, pred, f"{confidence:.2f}%", label])
            time.sleep(0.5)
        ConfusionMatrixDisplay.from_predictions(y_test, prediction)
        mpl.show()
except KeyboardInterrupt:
    print("\n\n=== INTERRUPTED ===")
    print(classification_report(y_test, prediction))
    with open(os.path.join(output_path, 'results.txt'), mode='w') as f:
        f.write(classification_report(y_test, prediction))
        print(f"False Positive Rating (FPR): {fp / neg}")
        print(f"False Negative Rating (FNR): {fn / pos}")
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        f.write(f"FPR: {fp / neg:.4f}, FNR: {fn / pos:.4f}\n")
    ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    mpl.show()

