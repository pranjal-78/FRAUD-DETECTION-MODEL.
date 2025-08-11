# CODE-HUT 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


df = pd.read_csv("Fraud.csv")


df_model = df.drop(["nameOrig", "nameDest"], axis=1)

df_model = pd.get_dummies(df_model, columns=["type"], drop_first=True)

df_sample = df_model.sample(100000, random_state=42)

X = df_sample.drop(["isFraud", "isFlaggedFraud"], axis=1)
y = df_sample["isFraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)


print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Fraud", "Fraud"],              #GRAPH
            yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(x=y, palette="coolwarm")
plt.title("Class Distribution Before SMOTE")
plt.show()
