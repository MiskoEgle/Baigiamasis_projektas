# Modelių Mokymo ir Testavimo Sistema

Ši sistema leidžia mokyti ir testuoti skirtingus modelius (KNN, CNN, Transformer) vaizdų klasifikavimo uždaviniams.

## Funkcionalumas

- Modelių mokymas su skirtingais duomenų rinkiniais
- Modelių testavimas su naujais paveikslėliais
- Rezultatų vizualizacija ir palyginimas
- Duomenų augmentacija
- Metrikų skaičiavimas ir atvaizdavimas

## Diegimas

1. Sukurkite virtualią aplinką:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Įdiekite reikalingas bibliotekas:
```bash
pip install -r requirements.txt
```

3. Sukonfigūruokite aplinkos kintamuosius:
```bash
cp .env.example .env
# Redaguokite .env failą pagal savo poreikius
```

4. Inicializuokite duomenų bazę:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Paleiskite aplikaciją:
```bash
flask run
```

## Naudojimas

1. Atidarykite naršyklę ir eikite į `http://localhost:5000`
2. Pasirinkite "Mokyti" arba "Testuoti" meniu
3. Sekite instrukcijas ekrane

## Projektų struktūra

```
app/
├── __init__.py
├── config.py
├── routes.py
├── models/
│   ├── __init__.py
│   ├── database.py
│   ├── knn_model.py
│   ├── cnn_model.py
│   └── transformer_model.py
├── utils/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── dataset_loader.py
│   └── visualization.py
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── templates/
    ├── index.html
    ├── mokyti.html
    ├── testuoti.html
    └── rezultatai.html
```

## Reikalavimai

- Python 3.8+
- Flask
- TensorFlow
- PyTorch
- OpenCV
- scikit-learn
- Kiti reikalavimai pateikti `requirements.txt` faile

## Licencija

Šis projektas licencijuojamas pagal MIT licenciją. Žr. `LICENSE` failą detalesnei informacijai. 