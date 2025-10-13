import os
import shutil

def organize_files(directory):
    for i in range(6):
        os.makedirs(os.path.join(directory, str(i)), exist_ok=True)

    for fichier in os.listdir(directory):
        if fichier.endswith(".png"):
            nom_sans_extension = fichier.split(".")[0]
            chiffre = nom_sans_extension[-2]
            if chiffre.isdigit():
                src = os.path.join(directory, fichier)
                dest = os.path.join(directory, chiffre, fichier)
                shutil.move(src, dest)
                print(f"Déplacé : {fichier} → dossier {chiffre}")
            else :
                print(f"Chiffre non numérique : {fichier}, le fichier n'a pas été déplacé.")
                
organize_files("../resources/dataset/archive/test")
organize_files("../resources/dataset/archive/train")
organize_files("../resources/dataset/archive/fingers/test")
organize_files("../resources/dataset/archive/fingers/train")
            
print("Opération terminée.")
