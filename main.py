import pandas as pd
from clustering import calcolaCluster
import os
from supervisedL import trainModelKFold




#PER INSTALLARE TUTTE LE LIBRERIE NECESSARIE FAI pip install -r requirements.txt    e non dovrebbe dare errori


#DATASET -> https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering



# Caricamento del dataset e selezione delle feature
directory = "dataset/OnlineRetail.csv"
dataset = pd.read_csv(directory, encoding='ISO-8859-1')
#Pre-processing : trasformo le colonne con valori stringa in valori numerici
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = pd.factorize(dataset[column])[0]

#Cancellazione delle righe con valori nulli
dataset.dropna(inplace=True)

etichette_cluster, centroidi = calcolaCluster(dataset)
#Creo il nuovo dataset con la colonna 'clusterIndex'
differentialColumn = 'clusterIndex'
dataset[differentialColumn] = etichette_cluster
model= trainModelKFold(dataset, differentialColumn)

#Ripeto l'esperimento eliminando le feature InvoiceNo, InvoiceDate, CustomerID  per vedere se il modello migliora
dataset=pd.read_csv(directory, encoding='ISO-8859-1')
dataset.drop(['InvoiceNo','InvoiceDate','CustomerID'], axis=1, inplace=True)
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = pd.factorize(dataset[column])[0]
dataset.dropna(inplace=True)
etichette_cluster, centroidi = calcolaCluster(dataset)
differentialColumn = 'clusterIndex'
dataset[differentialColumn] = etichette_cluster
model= trainModelKFold(dataset, differentialColumn)


