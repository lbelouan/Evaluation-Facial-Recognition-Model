import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

import time
import face_recognition

from deepface import DeepFace
from deepface.commons import functions

from deepface.basemodels import FbDeepFace, ArcFace, OpenFace, Facenet, VGGFace
# DeepID, SFace si vous souhaitez également tester ces modèles

# print("Téléchargement du modèle DeepFace...")
# deepface_model = FbDeepFace.load_model()

# print("Téléchargement du modèle ArcFace...")
# arcface_model = ArcFace.load_model()

# print("Téléchargement du modèle Facenet512d...")
# facenet512d_model = Facenet.load_facenet512d_model() 

# print("Téléchargement du modèle Facenet128d...")
# facenet128d_model = Facenet.load_facenet128d_model() 

print("Téléchargement du modèle OpenFace...")
openface_model = OpenFace.load_model()

# print("Téléchargement du modèle VGGFace...")
# vggface_model = VGGFace.load_model()

# print("Téléchargement du modèle DeepID...")
# deepid_model = DeepID.load_model()

# print("Téléchargement du modèle SFace...")
# sface_model = SFace.load_model()

model_dict = {
    # "DeepFace": deepface_model,
    # "ArcFace": arcface_model,
    # "Facenet512d": facenet512d_model
    # "Facenet128d": facenet128d_model,
    "OpenFace": openface_model,
    # "VGGFace": vggface_model
    # "DeepID": deepid_model
    # "SFace": sface_model
    
}

base_path="lfw_restricted" # on utilise une plus petite partie du dataset LFW
probes_path = os.path.join(base_path, "probes")
identite_path = os.path.join(base_path, "identite")

def get_embedding(img_path, model_name, model):
    try:
        

        model_name_mapped = {
            "DeepFace": "DeepFace",
            "ArcFace": "ArcFace",
            "Facenet512d": "Facenet",
            "Facenet128d": "Facenet",
            "VGGFace": "VGG-Face",
            "OpenFace": "OpenFace",
            "DeepID": "DeepID",
            "SFace": "SFace",
        }.get(model_name, "DeepFace")  # fallback sûr
        
        # 2. Mappage pour le type de normalisation
        normalization_type = {
            "DeepFace": "base",
            "ArcFace": "ArcFace",
            "Facenet512d": "Facenet",
            "Facenet128d": "Facenet",
            "VGGFace": "VGGFace",
            "OpenFace": "base",
            "DeepID": "base",
            "SFace": "base",
        }.get(model_name, "base") # cherche si trouve avec la clé model_name sinon retourne "base"
        
        # Étape 1 : redimensionner et détecter le visage
        target_size = functions.find_target_size(model_name_mapped)
        faces = functions.extract_faces(img=img_path, target_size=target_size)
        if len(faces) == 0:
            return None
        
        img_pixels = faces[0][0]
        
        img_pixels = functions.normalize_input(img_pixels, normalization=normalization_type)

        # Étape 3 : prédiction de l'embedding
        embedding = model.predict(img_pixels, verbose=0)[0].tolist()

        # 4. Normalisation L2        
        embedding = normalize([embedding])[0] 
        
        return embedding
    
    except Exception as e:
        import traceback
        print(f"[Ignoré] {img_path}")
        traceback.print_exc()
        return None


def comparer_images(nom_modele, chemin_img1, chemin_img2):
    
    if nom_modele == "Face_recognition":
        img1 = face_recognition.load_image_file(chemin_img1)
        img2 = face_recognition.load_image_file(chemin_img2)
        enc1 = face_recognition.face_encodings(img1)
        enc2 = face_recognition.face_encodings(img2)

        if len(enc1) == 0 or len(enc2) == 0:
            return None

        emb1 = enc1[0]
        emb2 = enc2[0]

    else:
        # Pour tous les modèles DeepFace
        emb1 = get_embedding(chemin_img1, nom_modele, model_dict[nom_modele])
        emb2 = get_embedding(chemin_img2, nom_modele, model_dict[nom_modele])

    if emb1 is None or emb2 is None or len(emb1) == 0 or len(emb2) == 0:
        return None

    # Distance euclidienne (fonctionne bien après normalisation L2)
    # print("Norme emb1 :", np.linalg.norm(emb1))
    # print("Norme emb2 :", np.linalg.norm(emb2))
    
    if nom_modele == "ArcFace":
        distance = cosine (emb1,emb2) # (valeurs entre -1 et 1) mais entre 0 et 1 pour des images

    else : 
        distance = np.linalg.norm(np.array(emb1) - np.array(emb2)) # distance est entre 0 et 2 
        distance = distance/2 # on normalise entre 0 et 1
    
    return distance

def convert_to_visual_format(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel() # intervertit les deux elements de la diagonale
    return np.array([[tp, fp],
                     [fn, tn]])

def evaluer_modele(nom_modele, seuil):
    """
    Évalue un modèle de reconnaissance faciale :
    - Calcule la distance pour chaque paire (identité, probe)
    - Compare au seuil
    - Calcule les métriques classiques
    - Mesure le temps total et moyen de reconnaissance
    """
    y_reel = []
    y_prevu = []
    temps_total = 0
    nb_comparaisons = 0
    
    personnes_probes = os.listdir(probes_path)
    personnes_identites = os.listdir(identite_path)
    
    for personne_probe in personnes_probes: # boucle sur les personnes du dossier probes
        chemin_probes = os.path.join(probes_path, personne_probe)
        images_probes = [os.path.join(chemin_probes, f) for f in os.listdir(chemin_probes)] # on stock les 3 images probes
        print(f"----- Personne : {personne_probe}")
        
        for img_probe in images_probes: # boucle dans nos 3 images probes
            for personne_id in personnes_identites: # boucle sur les personnes du dossier identite
                chemin_id = os.path.join(identite_path, personne_id)
                img_id = os.path.join(chemin_id, os.listdir(chemin_id)[0])  # on stocke l'image d'identité

                debut = time.time()
                distance = comparer_images(nom_modele, img_id, img_probe)
                fin = time.time() 
                

                if distance is None:
                        continue # saute le reste de la boucle en cours

                prediction = int(distance < seuil)
                
                if prediction :
                    print(f"Distance gardée: {distance}")
                
                vrai_label = int(personne_probe == personne_id) # si le nom des personnes est le même alors les imgs probes correspondent à l'image identité
                
                y_reel.append(vrai_label) # label vrai
                y_prevu.append(prediction) # label prédit 

                temps_total += (fin - debut)
                nb_comparaisons += 1
                

    matrice = confusion_matrix(y_reel, y_prevu) # Matrice de confusion sous ce format ([TN, FP], [FN, TP]) 
    matrice_conv=convert_to_visual_format(matrice) # Nécéssité d'inverser les élements de la diagonale(TP et TN)
    metriques = {
        "Exactitude (Accuracy)": accuracy_score(y_reel, y_prevu),
        "Précision": precision_score(y_reel, y_prevu),
        "Rappel (Recall)": recall_score(y_reel, y_prevu),
        "F1-Score": f1_score(y_reel, y_prevu),
        "Nombre de comparaisons": nb_comparaisons,
        "Temps total (s)": temps_total,
        "Temps moyen (s)": temps_total / nb_comparaisons  if nb_comparaisons>0 else 0
    }
    print(f"Nombre total de comparaisons : {len(y_reel)}")
    print(f"Nombre de positifs attendus (sum y_reel) : {sum(y_reel)}")
    print(f"Nombre de prédits positifs (sum y_prevu) : {sum(y_prevu)}")

    print(y_reel)
    print(y_prevu)
    
    return matrice_conv, metriques

# === Exemple d’utilisation ===
if __name__ == "__main__":
    
    modele = "OpenFace" 
    
    if modele == "Face_recognition" :
        seuil=0.3
        
    elif modele == "ArcFace" :
        seuil=0.61
        
    elif modele == "DeepFace" :
        seuil=0.33 

    elif modele == "VGGFace" : # correspond à VGGFace 2
        seuil=0.6
        
    elif modele == "OpenFace" :
        seuil=0.37

    elif modele == "Facenet512d" :
        seuil=0.47
        
    elif modele == "Facenet128d" :
        seuil=0.5
    
    elif modele == "DeepID" :
        seuil=0.35
            
    elif modele == "SFace" :
        seuil=0.35    
    
    print(seuil)
    
    matrice, resultats = evaluer_modele(modele, seuil)

    print(f"\n=== Résultats pour le modèle : {modele} ===")
    print("Matrice de confusion :\n", matrice)
    for cle, val in resultats.items():
        print(f"{cle} : {val:.4f}" if isinstance(val, float) else f"{cle} : {val}")
