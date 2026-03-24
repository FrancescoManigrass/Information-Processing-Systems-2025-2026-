import os

# Percorso della cartella contenente le immagini
dir_path = "mybottle/images/"

# Ottieni la lista dei file .jpg e ordina per nome (numerico, anche se ci sono buchi)
files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
files.sort(key=lambda x: int(os.path.splitext(x)[0]))

# Rinomina temporaneamente per evitare conflitti
for i, filename in enumerate(files):
    temp_name = f"temp_{i}.jpg"
    os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, temp_name))

# Rinomina finale in ordine sequenziale 0.jpg, 1.jpg, ...
temp_files = [f for f in os.listdir(dir_path) if f.startswith('temp_') and f.endswith('.jpg')]
temp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
for i, filename in enumerate(temp_files):
    new_name = f"{i}.jpg"
    os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_name))

print(f"Immagini rinominate in ordine sequenziale da 0.jpg a {len(temp_files)-1}.jpg")
