from pathlib import Path

# File di input
input_file = Path("captions_risultato.txt")

# Cartella di output
output_dir = Path("righe_divise")
output_dir.mkdir(exist_ok=True)

# Legge il file e crea un file nuovo per ogni riga
with input_file.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        output_file = output_dir / f"{i}.txt"
        with output_file.open("w", encoding="utf-8") as out:
            out.write(line.rstrip("\n"))

print(f"Creati {i} file nella cartella: {output_dir}")