import os

# Percorso della cartella contenente le immagini
DIR_PATH = "dataset_bottle"

# Filtra tutti i file immagine (jpg, jpeg, png)
image_exts = {'.jpg', '.jpeg', '.png'}
image_files = [f for f in os.listdir(DIR_PATH) if os.path.splitext(f)[1].lower() in image_exts]

# Ordina alfabeticamente (puoi cambiare in base a data/altro criterio se serve)
image_files.sort()

# Rinomina i file da 0.jpg, 1.jpg, ...
for i, filename in enumerate(image_files):
    new_name = f"{i}.jpg"
    src = os.path.join(DIR_PATH, filename)
    dst = os.path.join(DIR_PATH, new_name)
    if src != dst:
        os.rename(src, dst)
print(f"Rinominati {len(image_files)} file da 0.jpg a {len(image_files)-1}.jpg")
