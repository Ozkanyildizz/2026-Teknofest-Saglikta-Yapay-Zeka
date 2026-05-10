import csv

with open("AlphaMissense_aa_substitutions.tsv", "r", newline="", encoding="utf-8") as tsv_file:
    reader = csv.reader(tsv_file, delimiter="\t")
    with open("AlphaMissense_aa_substitutions.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for row in reader:
            writer.writerow(row)