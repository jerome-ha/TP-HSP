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

### Kernels CUDA et Fonctions Nécessaires

1. **Conv2D**: Implémenter un kernel pour la convolution 2D.
2. **AveragePooling2D**: Implémenter un kernel pour le sous-échantillonnage moyen.
3. **Dense**: Implémenter un kernel pour la multiplication matricielle et l'addition du biais.
4. **Activation Functions**: Utiliser `tanhf` pour `tanh` et implémenter un kernel pour `softmax`.

### Importation des Poids et Biais Pré-Entraînés

- Les poids et biais pré-entraînés doivent être importés et transférés sur le GPU avant de lancer les kernels.

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

## Remarques

- Cette implémentation suppose une familiarité avec la programmation CUDA et les concepts de base des réseaux de neurones.
- Les détails tels que la gestion de la mémoire, la synchronisation des opérations et la gestion des erreurs sont cruciaux pour une implémentation réussie.
- L'optimisation des performances peut nécessiter des ajustements selon les spécificités du matériel GPU utilisé.
