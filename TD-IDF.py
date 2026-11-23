import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

projectPath = Path(__file__).parent

manifestPath = projectPath / "training set" / "dataset" / "manifest_all_complete.csv"

datasetRoot = manifestPath.parent

df = pd.read_csv(manifestPath)

df["text"] = df["relative_path"].apply(
    lambda path: (datasetRoot / path).read_text(encoding="utf-8-sig")
)

trainDf = df[df["split"] == "train"]
valDf = df[df["split"] == "validation"]
testDf = df[df["split"] == "test"]

XTrainText = trainDf["text"]
XValText = valDf["text"]
XTestText = testDf["text"]

yTrain = trainDf["class_id"]
yVal = valDf["class_id"]
yTest = testDf["class_id"]

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    token_pattern=r"(?u)\b[a-zA-Z]+\b"
)

XTrain = tfidf.fit_transform(XTrainText)
XVal= tfidf.transform(XValText)
XTest = tfidf.transform(XTestText)
print("\nTF-IDF Matrix Shapes:")
print("Training:", XTrain.shape)
print("Validation:", XVal.shape)
print("Testing:", XTest.shape)

print("\nFirst 20 vocabulary terms:")
print(tfidf.get_feature_names_out()[:20])

classifier = LogisticRegression(
    max_iter=2000,
    C=1.0
)

classifier.fit(XTrain, yTrain)

valPredictions = classifier.predict(XVal)

print("Classification Report:")
print(classification_report(
    yVal,
    valPredictions,
    target_names=["Human", "Light AI", "Heavy AI", "Full AI"]
))

print("Validation Macro F1:", f1_score(yVal,valPredictions,average="macro"))

print("\nConfusion Matrix:")
cm = confusion_matrix(yVal, valPredictions)
print(cm)

classNames = ["Human", "Light AI", "Heavy AI", "Full AI"]

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classNames,
    yticklabels=classNames
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("TF-IDF Logistic Regression Confusion Matrix")

plt.tight_layout()
plt.show()
