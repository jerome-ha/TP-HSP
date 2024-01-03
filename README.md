# CNN_implementation_on_CUDA
# Implémentation d'un Modèle LeNet-5 sur CUDA

Ce README fournit un guide pour implémenter le modèle de réseau de neurones LeNet-5 sur CUDA en utilisant des poids et des biais pré-entraînés.

## Structure du Modèle

Le modèle LeNet-5 comprend les couches suivantes :

1. **Conv2D (C1)**: Convolution avec 6 filtres de taille 5x5, activation `tanh`.
2. **AveragePooling2D (S2)**: Sous-échantillonnage moyen.
3. **Conv2D (C3)**: Convolution avec 16 filtres de taille 5x5, activation `tanh`.
4. **AveragePooling2D (S4)**: Sous-échantillonnage moyen.
5. **Flatten**: Aplatit les données en un vecteur unidimensionnel.
6. **Dense (C5)**: Couche dense avec 120 unités, activation `tanh`.
7. **Dense (F6)**: Couche dense avec 84 unités, activation `tanh`.
8. **Dense (Output Layer)**: Couche de sortie avec 10 unités, activation `softmax`.

## Implémentation CUDA

### Kernels CUDA et Fonctions Nécessaires (dans layers.cu)
0. Avant d'implémenter les differents layers, on se familiarise avec CUDA en implémentant diverses opérations nécessaire au matrice. Tout est résumé sur mat_mulp.cu
1. **Conv2D**: Implémenter un kernel pour la convolution 2D.
2. **AveragePooling2D**: Implémenter un kernel pour le sous-échantillonnage moyen.
3. **Dense**: Implémenter un kernel pour la multiplication matricielle et l'addition du biais.
4. **Activation Functions**: Utiliser `tanhf` pour `tanh` et implémenter un kernel pour `softmax` (assez avancé pas fait).

Remarque : on travail déjà avec des données unidimentionnelles donc flatten n'est pas utile.

### Importation des Poids et Biais Pré-Entraînés

- Les poids et biais pré-entraînés sur le fichier LeNet5.ipynb qui donne les fichers contenant les poids et biais FashionMNIST_weights.h5 doivent être importés et transférés sur le GPU avant de lancer les kernels, cela remplacent les matrices random initiées.

### Configuration de Grid et de Block

- Choisir les dimensions de `dimBlock` et `dimGrid` en fonction de la taille des données et des capacités du GPU.
- Une configuration optimale est essentielle pour maximiser les performances.

## Avantages de l'Utilisation du GPU

### Parallélisme Massif

- Les GPU offrent un parallélisme massif, rendant le traitement de réseaux de neurones profonds beaucoup plus rapide que sur CPU.

### Efficacité

- Les opérations telles que les convolutions et les multiplications matricielles sont particulièrement bien adaptées au modèle de calcul du GPU.

### Utilisation Pratique

- Cette implémentation est idéale pour l'inférence rapide en utilisant des modèles pré-entraînés, profitant de la vitesse de traitement du GPU.

### Temps gagnés en pratique 

Il faudrait réimplémenter le model en CPU et mesurer pour chaque le temps de calcul, mais on peut faire un estimation car c'est le code qui détermine la parallélisation des calculs sur CUDA, Calculons alors le nombre de threads sur notre implémentation de conv2D:

dimBlock est configuré à (6, 6). Cela signifie que chaque bloc contient
6×6=36 threads. dimGrid est calculé comme ((28 + 5) / 6, (28 + 5) / 6, 6),

Calculons dimGrid :

La largeur et la hauteur de dimGrid seront  ( 28+5)/6=33/6 (28+5)/6=33/6, ce qui donne 5.5. En CUDA, le nombre de blocs est toujours un entier, donc cela sera arrondi à 6 blocs dans chaque dimension (x et y).
La profondeur de dimGrid (dimension z) est 6, car vous avez 6 noyaux de convolution.
Ainsi, dimGrid est approximativement (6, 6, 6).

Le nombre total de blocs dans la grille est donc
6×6×6=216 blocs.

#Le nombre total de threads qui peuvent être exécutés en parallèle est donc approximativement 36(threads par bloc)×216(blocs)= ##7776 threads.

Le code pourrait théoriquement rendre l'execution 7000 fois plus rapide. En pratique, le nombre réel de threads exécutés en parallèle peut être limité par les capacités matérielles du GPU, telles que le nombre de cœurs CUDA et la quantité de mémoire disponible.

## Remarques

- Cette implémentation suppose une familiarité avec la programmation CUDA et les concepts de base des réseaux de neurones.
- Les détails tels que la gestion de la mémoire, la synchronisation des opérations et la gestion des erreurs sont cruciaux pour une implémentation réussie.
- L'optimisation des performances est différente selon les spécificités du matériel GPU utilisé, le temps de calcul mesuré sur mat_mulp.cu n'est pas le même avec une config PC différente.
