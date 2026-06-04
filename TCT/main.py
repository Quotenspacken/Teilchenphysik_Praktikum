import runpy

skripte = [
    "Auswertung_Bias_vorderseite.py",
    "Auswertung_Laser_vorderseite.py",
    "Auswertung_Bias_rueckseite.py",
    "Auswertung_Laser_rueckseite.py",
]

for skript in skripte:
    print(f"\n{'='*50}")
    print(f"Starte: {skript}")
    print(f"{'='*50}\n")
    runpy.run_path(skript, run_name="__main__")
    print(f"\nFertig: {skript}")

print("\nAlle Auswertungen abgeschlossen.")
