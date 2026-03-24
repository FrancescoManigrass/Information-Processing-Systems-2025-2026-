import os

# Percorso della cartella contenente le immagini del dataset
DIR_PATH = "dataset_bottle/"

# Filtra solo i file con nome numerico (es: 0001.jpg, 0023.jpg, ecc.)
numeric_files = []
for f in os.listdir(DIR_PATH):
    name, ext = os.path.splitext(f)
    if ext.lower() == '.jpg' and name.isdigit():
        numeric_files.append(f)

# Ordina i file numerici per valore intero
numeric_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

# Rinomina temporaneamente per evitare conflitti
for i, filename in enumerate(numeric_files):
    temp_name = f"temp_{i}.jpg"
    os.rename(os.path.join(DIR_PATH, filename), os.path.join(DIR_PATH, temp_name))

# Rinomina finale in ordine sequenziale 0.jpg, 1.jpg, ...
temp_files = [f for f in os.listdir(DIR_PATH) if f.startswith('temp_') and f.endswith('.jpg')]
temp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
for i, filename in enumerate(temp_files):
    new_name = f"{i}.jpg"
    os.rename(os.path.join(DIR_PATH, filename), os.path.join(DIR_PATH, new_name))

print(f"Immagini numeriche in {DIR_PATH} rinominate in ordine sequenziale da 0.jpg a {len(temp_files)-1}.jpg")
