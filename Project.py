# Importation des packages nécessaires
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
from matplotlib import gridspec

# Chargement du jeu de données depuis le fichier CSV en utilisant pandas
# La meilleure méthode est de monter le lecteur sur Colab et
# de copier le chemin d'accès du fichier CSV
data = pd.read_csv("creditcard.csv")

# Jeter un coup d'œil aux données
data.head(10)

# Afficher la forme des données
print(data.shape)
print(data.describe())

# Matrice de corrélation
corrmat = data.corr()

plt.figure(figsize=(16, 10))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.show()

# Créer un échantillon aléatoire pour éviter une visualisation trop lourde
sample_data = data.sample(frac=0.03, random_state=42)
# Histogramme du temps de transaction pour les classes "Valide" et "Fraude"
plt.figure(figsize=(12, 6))
sns.histplot(data[data['Class'] == 0]['Time'], kde=False, label='Transaction Valide')
sns.histplot(data[data['Class'] == 1]['Time'], kde=False, label='Fraude')
plt.xlabel('Temps de la transaction')
plt.ylabel('Nombre d\'occurrences')
plt.legend()
plt.title('Distribution temporelle des transactions')
plt.show()

# Diagramme à barres de la distribution des classes
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=data)
plt.title('Distribution des classes')
plt.xlabel('Classe')
plt.ylabel('Nombre d\'occurrences')
plt.show()

# Boîtes à moustaches pour les montants de transactions avec distinction des classes
plt.figure(figsize=(14, 8))
sns.boxplot(x='Class', y='Amount', data=data)
plt.yscale('log')  # Pour mieux visualiser la plage des montants
plt.title('Boîtes à moustaches pour les montants de transactions')
plt.show()

# Calcul du nombre de cas de fraude dans le jeu de données
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
print('Amount details of the fraudulent transaction')
fraud.Amount.describe()
print('details of valid transaction')
valid.Amount.describe()

# Gestion des valeurs manquantes
data.fillna(data.mean(), inplace=True)

# Normalisation des caractéristiques numériques
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Division des X et Y du jeu de données
X = sample_data.drop(['Class'], axis=1)
Y = sample_data["Class"]
print(X.shape)
print(Y.shape)

# Obtenir uniquement les valeurs pour le traitement
# (c'est un tableau numpy sans colonnes)
xData = X.values
yData = Y.values

# Utilisation de Scikit-learn pour diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
# Diviser les données en ensembles d'entraînement et de test
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# Création du modèle de classification Random Forest
from sklearn.ensemble import RandomForestClassifier
# Création du modèle de forêt aléatoire
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Prédictions
yPred = rfc.predict(xTest)

# Évaluation du classificateur
# Impression de chaque score du classificateur
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

#n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("Le modèle utilisé est le classificateur de la forêt aléatoire")

acc = accuracy_score(yTest, yPred)
print("L'exactitude est {}".format(acc))

prec = precision_score(yTest, yPred)
print("La précision est {}".format(prec))

rec = recall_score(yTest, yPred)
print("Le rappel est {}".format(rec))

f1 = f1_score(yTest, yPred)
print("Le F1-Score est {}".format(f1))

# Impression de la matrice de confusion
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
print(conf_matrix)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
			yticklabels=LABELS, annot=True, fmt="d")
plt.title("Matrice de confusion")
plt.ylabel('Classe réelle')
plt.xlabel('Classe prédite')
plt.show()

# Calculer les probabilités prédites
#yProb = rfc.predict_proba(xTest)[:, 1]

# Création du modèle de régression logistique
logistic_model = LogisticRegression()
logistic_model.fit(xTrain, yTrain)

# Prédictions
yPred_logistic = logistic_model.predict(xTest)

# Évaluation du modèle de régression logistique
acc_logistic = accuracy_score(yTest, yPred_logistic)
print("L'exactitude de la régression logistique est {}".format(acc_logistic))

prec_logistic = precision_score(yTest, yPred_logistic)
print("La précision de la régression logistique est {}".format(prec_logistic))

rec_logistic = recall_score(yTest, yPred_logistic)
print("Le rappel de la régression logistique est {}".format(rec_logistic))

f1_logistic = f1_score(yTest, yPred_logistic)
print("Le F1-Score de la régression logistique est {}".format(f1_logistic))
# Impression de la matrice de confusion pour la régression logistique
conf_matrix_logistic = confusion_matrix(yTest, yPred_logistic)
print(conf_matrix_logistic)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix_logistic, xticklabels=LABELS,yticklabels=LABELS, annot=True, fmt="d")
plt.title("Matrice de confusion - Régression logistique")
plt.ylabel('Classe réelle')
plt.xlabel('Classe prédite')
plt.show()

# Optimisation des paramètres pour la forêt aléatoire
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=4)
grid_search_rf.fit(xTrain, yTrain)

# Affichage des meilleurs paramètres

print("Meilleurs paramètres pour la forêt aléatoire :", grid_search_rf.best_params_)

# Optimisation des paramètres pour la régression logistique
param_grid_logistic = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

logistic_model = LogisticRegression()
grid_search_logistic = GridSearchCV(estimator=logistic_model, param_grid=param_grid_logistic, cv=4)
grid_search_logistic.fit(xTrain, yTrain)

# Affichage des meilleurs paramètres
print("Meilleurs paramètres pour la régression logistique :", grid_search_logistic.best_params_)
