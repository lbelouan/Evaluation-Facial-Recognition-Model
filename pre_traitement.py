import os
import shutil

def clean_lfw_dataset(dataset_path):
    """
    Nettoie le dataset LFW :
    - Supprime les dossiers contenant strictement moins de 4 images
    - Conserve seulement les 4 premières images dans chaque dossier restant
    """
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        images = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        num_images = len(images)

        if num_images < 4:
            # Supprimer le dossier complet
            shutil.rmtree(person_path)
        else:
            # Garder seulement les 4 premières
            for image_to_remove in images[4:]:
                os.remove(os.path.join(person_path, image_to_remove))

# Exemple d'utilisation :
#clean_lfw_dataset("lfw")

def move_first_image_to_identite(base_path):
    probes_path = os.path.join(base_path, "probes")
    identite_path = os.path.join(base_path, "identite")

    for person_name in os.listdir(probes_path):
        person_dir = os.path.join(probes_path, person_name)

        if not os.path.isdir(person_dir):
            continue
        for file_name in os.listdir(person_dir):
            
            src_image_path = os.path.join(person_dir, file_name)
            # Créer le dossier identite/person_name
            dest_person_dir = os.path.join(identite_path, person_name)
            os.makedirs(dest_person_dir, exist_ok=True)
            
            dest_image_path = os.path.join(dest_person_dir, file_name)
            
            shutil.move(src_image_path, dest_image_path)
            print(f"Déplacé : {src_image_path} → {dest_image_path}")
            break  # une seule image suffit

move_first_image_to_identite("lfw")