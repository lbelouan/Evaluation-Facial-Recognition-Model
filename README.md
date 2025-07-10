# Évaluation de Modèles de Deep Learning pour la Reconnaissance Faciale

Ce projet permet d'évaluer et de comparer différents modèles de reconnaissance faciale sur le dataset LFW (Labeled Faces in the Wild). Il inclut des outils de prétraitement des données et d'évaluation complète des performances.

## Table des Matières

- [Description du Projet](#description-du-projet)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles Supportés](#modèles-supportés)
- [Métriques d'Évaluation](#métriques-dévaluation)
- [Optimisation](#optimisation)
- [Troubleshooting](#troubleshooting)

## Description du Projet

Ce projet vise à évaluer les performances de différents modèles de reconnaissance faciale en utilisant le dataset LFW. Il permet de comparer plusieurs modèles de deep learning, calculer des métriques de performance standardisées, mesurer les temps de traitement et organiser automatiquement les données d'entraînement et de test.

## Structure du Projet

```
Evaluation_modele_deep_learning/
├── evaluate_model.py          # Script principal d'évaluation
├── pre_traitement.py          # Script de prétraitement des données
├── requirements.txt           # Dépendances Python
├── lfw/                      # Dataset LFW complet
│   ├── identite/             # Images d'identité (1 par personne)
│   └── probes/               # Images de test (3 par personne)
├── lfw_restricted/           # Sous-ensemble du dataset LFW
│   ├── identite/             # Images d'identité filtrées
│   └── probes/               # Images de test filtrées
└── venv/                     # Environnement virtuel Python
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'Installation

1. **Cloner le repository** :
   ```bash
   git clone <url-du-repository>
   cd Evaluation_modele_deep_learning
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv venv
   ```

3. **Activer l'environnement virtuel** :
   - Windows :
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac :
     ```bash
     source venv/bin/activate
     ```

4. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   
5. **Télécharger le dataset LFW People** :
   [lien du Dataset](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)

**Note** : L'installation peut prendre plusieurs minutes car certains modèles (comme dlib) nécessitent une compilation.

## Utilisation

### Prétraitement des Données (`pre_traitement.py`)

Ce script prépare le dataset LFW pour l'évaluation :

- **`clean_lfw_dataset(dataset_path)`** : Nettoie le dataset en supprimant les dossiers avec moins de 4 images
- **`move_first_image_to_identite(base_path)`** : Organise les images en séparant identité et probes

```python
# Nettoyer le dataset complet
clean_lfw_dataset("lfw")

# Organiser les données pour l'évaluation
move_first_image_to_identite("lfw")
```

### Évaluation des Modèles (`evaluate_model.py`)

Ce script principal évalue les performances des modèles de reconnaissance faciale.

#### Configuration

Modifiez les paramètres suivants dans le script :

```python
# Sélection du modèle à évaluer
modele = "OpenFace"  # Options disponibles dans model_dict

# Seuils de décision par modèle
seuils = {
    "Face_recognition": 0.3,
    "ArcFace": 0.61,
    "DeepFace": 0.33,
    "VGGFace": 0.6,
    "OpenFace": 0.37,
    "Facenet512d": 0.47,
    "Facenet128d": 0.5,
    "DeepID": 0.35,
    "SFace": 0.35
}
```

#### Exécution

```bash
python evaluate_model.py
```

## Modèles Supportés

| Modèle | Description | Seuil Recommandé |
|--------|-------------|------------------|
| **OpenFace** | Modèle open source de reconnaissance faciale | 0.37 |
| **ArcFace** | Modèle avec perte additive angulaire | 0.61 |
| **DeepFace** | Modèle Facebook de reconnaissance faciale | 0.33 |
| **VGGFace** | Modèle basé sur l'architecture VGG | 0.6 |
| **Facenet512d** | FaceNet avec embeddings 512D | 0.47 |
| **Facenet128d** | FaceNet avec embeddings 128D | 0.5 |
| **DeepID** | Modèle de reconnaissance d'identité profonde | 0.35 |
| **SFace** | Modèle de reconnaissance faciale rapide | 0.35 |
| **Face_recognition** | Bibliothèque Python simple | 0.3 |

## Métriques d'Évaluation

Le script calcule automatiquement les métriques suivantes :

### Métriques de Performance
- **Exactitude (Accuracy)** : Pourcentage de prédictions correctes
- **Précision** : Proportion de vrais positifs parmi les prédictions positives
- **Rappel (Recall)** : Proportion de vrais positifs détectés
- **F1-Score** : Moyenne harmonique de la précision et du rappel

### Métriques de Temps
- **Temps total** : Durée totale de l'évaluation
- **Temps moyen** : Temps moyen par comparaison

### Matrice de Confusion
Format : `[[TP, FP], [FN, TN]]`
- **TP** : Vrais Positifs
- **FP** : Faux Positifs  
- **FN** : Faux Négatifs
- **TN** : Vrais Négatifs

## Optimisation

### Chargement Sélectif des Modèles

Pour tester un seul modèle et éviter de recharger ou télécharger tous les modèles, commentez les lignes correspondantes dans `evaluate_model.py` :

```python
# Exemple : pour tester uniquement OpenFace
print("Téléchargement du modèle OpenFace...")
openface_model = OpenFace.load_model()

# Commenter les autres modèles
# print("Téléchargement du modèle DeepFace...")
# deepface_model = FbDeepFace.load_model()

# print("Téléchargement du modèle ArcFace...")
# arcface_model = ArcFace.load_model()

model_dict = {
    "OpenFace": openface_model,
    # "DeepFace": deepface_model,  # Commenté
    # "ArcFace": arcface_model,    # Commenté
}
```

### Optimisation DeepFace

Au lieu d'utiliser directement l'API DeepFace qui recharge le modèle à chaque comparaison, nous utilisons les fonctions de la librairie pour optimiser la pipeline :

```python
def get_embedding(img_path, model_name, model):
    # Utilisation directe des fonctions DeepFace pour éviter de recharger le modèle
    model_name_mapped = {
        "DeepFace": "DeepFace",
        "ArcFace": "ArcFace",
        "Facenet512d": "Facenet",
        # ... autres modèles
    }.get(model_name, "DeepFace")
    
    # Détection et prétraitement optimisés
    target_size = functions.find_target_size(model_name_mapped)
    faces = functions.extract_faces(img=img_path, target_size=target_size)
    
    # Normalisation adaptée au modèle
    normalization_type = {
        "DeepFace": "base",
        "ArcFace": "ArcFace",
        # ... autres types
    }.get(model_name, "base")
    
    img_pixels = functions.normalize_input(faces[0][0], normalization=normalization_type)
    
    # Prédiction directe avec le modèle déjà chargé
    embedding = model.predict(img_pixels, verbose=0)[0].tolist()
    embedding = normalize([embedding])[0]
    
    return embedding
```

**Avantages de cette approche :**
- Le modèle est chargé une seule fois en mémoire
- Évite les appels répétés à l'API DeepFace
- Réduction significative du temps de calcul
- Contrôle précis du pipeline de traitement

### Autres Optimisations

1. **Utilisation du GPU** : Les modèles TensorFlow bénéficient automatiquement du GPU si disponible
2. **Sous-ensemble du dataset** : Utilisez `lfw_restricted` au lieu du dataset complet
3. **Cache des embeddings** : Implémenter un système de cache pour éviter de recalculer les embeddings

## Troubleshooting

### Problèmes Courants

#### Erreur d'Installation de dlib
```bash
# Windows - Installer Visual Studio Build Tools
# Linux - Installer les dépendances système
sudo apt-get install cmake
sudo apt-get install libopenblas-dev liblapack-dev
```

#### Erreur de Mémoire
- Réduire la taille du dataset en utilisant `lfw_restricted`
- Fermer les autres applications gourmandes en mémoire

#### Visages Non Détectés
- Vérifier la qualité des images
- Ajuster les paramètres de détection dans `functions.extract_faces()`

#### Temps de Calcul Trop Long
- Utiliser un sous-ensemble du dataset
- Désactiver les modèles non nécessaires dans `model_dict`

## Notes Importantes

- Les seuils fournis sont des valeurs de référence et peuvent nécessiter un ajustement selon votre cas d'usage
- Le dataset LFW est automatiquement téléchargé lors de la première utilisation
- Les modèles sont téléchargés automatiquement lors de leur première utilisation
- Les résultats peuvent varier selon la qualité des images et les conditions d'éclairage

## Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche pour votre fonctionnalité
3. Ajouter des tests si nécessaire
4. Soumettre une pull request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 
