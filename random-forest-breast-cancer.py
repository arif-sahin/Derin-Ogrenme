import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Reading the data
df = pd.read_csv('data - data.csv')
df = df.drop('id', axis=1)

# Eğitim ve test için veriyi ayırma
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis=1), df['diagnosis'], test_size=0.2, random_state=42)

# Random Forest
ran_for = RandomForestClassifier()
ran_for.fit(X_train, y_train)

# Tahmin
y_pred = ran_for.predict(X_test)

# Modeli Değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='B')
recall = recall_score(y_test, y_pred, pos_label='B')
f1 = f1_score(y_test, y_pred, pos_label='B')



print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Random Forest Precision: {precision:.4f}")
print(f"Random Forest Recall: {recall:.4f}")
print(f"Random Forest f1: {f1:.4f}")

cm1 = confusion_matrix(y_test ,y_pred)
#Confusion Matrix Isı Haritası
plt.figure(figsize=(4,3))
plt.rcParams.update({'font.size': 16})
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=ran_for.classes_)
disp.plot(cmap='Greens')
