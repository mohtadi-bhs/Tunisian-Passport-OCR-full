# compte_dataset.py : à lancer depuis "C:\Users\User\Documents\Projet par défaut"
import csv, collections, pathlib, re

ROOT = pathlib.Path("dataset_new") if pathlib.Path("dataset_new").exists() else pathlib.Path("data_aug_new")
NAMES = ["address_ar","date_of_birth","date_of_expiry","date_of_issue","full_name_ar","given_names",
         "identity_number","issuing_place","job_ar","nationality","passport_number","place_of_birth",
         "place_of_birth_ar","sex","surname"]

print(f"Dossier analysé : {ROOT.resolve()}\n")
print("| Id | Classe | Train | Validation |")
print("|---|---|---|---|")
counts = {}
for sp in ("train", "val"):
    c = collections.Counter()
    for f in (ROOT / "labels" / sp).glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    counts[sp] = c
for i, n in enumerate(NAMES):
    print(f"| {i} | {n} | {counts['train'][i]} | {counts['val'][i]} |")

for sp in ("train", "val"):
    imgs = list((ROOT / "images" / sp).glob("*"))
    bases = {re.sub(r"_aug\d+$", "", p.stem) for p in imgs}
    print(f"\n{sp} : {len(imgs)} images, {len(bases)} photos d'origine")

# modèle du jeu de test
with open("modele_test.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["photo","passportNumber","surname","givenNames","nameArabic","nationality",
                "dateOfBirth","sex","placeOfBirth","dateOfIssue","dateOfExpiry","issuingPlace",
                "identityNumber","addressArabic","jobArabic","mrz_valide"])
print("\nFichier créé : modele_test.csv (à remplir avec la vérité terrain de chaque photo de test)")