"""Bozuk CSV dosyasını düzelt - her satırdaki sarmalayan tırnakları kaldır."""
import pandas as pd

with open("missense_dataset.csv", "r") as f:
    content = f.read()

lines = content.strip().split("\n")
cleaned = []
for line in lines:
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    cleaned.append(line)

with open("missense_dataset.csv", "w", newline="") as f:
    f.write("\n".join(cleaned) + "\n")

# Dogrulama
df = pd.read_csv("missense_dataset.csv")
print("Sutunlar:", list(df.columns))
print("Satir sayisi:", len(df))
print(df.head(2))
print("\nCSV duzeltildi!")
