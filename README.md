# Automatische Klassifikation von Kundenanfragen

Ein KI-gestütztes System zur automatischen Klassifikation und Informationsextraktion aus Kunden-E-Mails im medizinischen Abrechnungsbereich.

## 📋 Funktionsübersicht

Das Tool erfüllt folgende Aufgaben:

1. **Klassifikation** von Kundenanfragen in 8 Kategorien
2. **Extraktion von Kundennummern** (Format: `X-XXXXX-XXXXXXXX`)
3. **Extraktion personenbezogener Daten** (Name, Geburtsdatum, Anschrift, Rechnungsbetrag)
4. **Kategoriespezifische Detailextraktion** (Ratenhöhe, Zahlungsziel, etc.)
5. **Strukturierte JSON-Ausgabe** für Weiterverarbeitung
6. **Evaluation** mit Confusion Matrix und Metriken

## 🏗️ Projektstruktur

```
├── src/
│   ├── model.py          # Klassifikation, Extraktion, LLM-Integration
│   ├── evaluation.py     # Batch-Verarbeitung, Metriken, Visualisierungen
│   └── prompts.py        # Prompt-Templates für LLM
├── data/
│   ├── data.csv                    # Eingabedaten
│   └── classification_targets.txt  # Zielkategorien
├── output/
│   ├── confusion_matrix.png        # Konfusionsmatrix (Heatmap)
│   ├── confusion_matrix.csv        # Konfusionsmatrix (CSV)
│   ├── class_distribution.png      # Klassenverteilung
│   ├── metrics_per_class.png       # Precision/Recall/F1 pro Klasse
│   ├── all_predictions.json        # Detaillierte Vorhersagen
│   └── predictions_full.csv        # Vollständige Vorhersagen (CSV)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Installation & Ausführung

### Voraussetzungen

- Python 3.10+
- [Ollama](https://ollama.ai/) mit `llama3` Modell

### Setup

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt
```

#### Option A: Ollama lokal installieren

```bash
# Ollama installieren (https://ollama.ai/)
# Dann Modell laden:
ollama pull llama3
ollama serve
```

#### Option B: Ollama via Docker

```bash
docker-compose up -d
# Wartet automatisch bis llama3 geladen ist
```

### Daten bereitstellen

Die Eingabedaten müssen als CSV-Datei unter `data/data.csv` abgelegt werden. Erforderliche Spalten:

| Spalte | Beschreibung |
|--------|-------------|
| `Betreff` | Betreffzeile der E-Mail |
| `Text` | Inhalt der E-Mail |
| `Anlagen` | Anhänge (optional) |
| `Anliegen` | Ground-Truth-Label für Evaluation |

### Ausführung

```bash
# Komplette Evaluation mit Visualisierungen
python src/evaluation.py
```

## 🔧 Technischer Ansatz

### Gewählte Methode: LLM mit Structured Output

Das System nutzt **Llama 3** (lokal via Ollama) mit LangChain für:
- **Zero-Shot Classification** durch sorgfältiges Prompt Engineering
- **Structured Output** via Pydantic-Schemas für typsichere Extraktion

### Architektur (Zweistufiger Prozess)

```
┌────────────────────┐     ┌──────────────────────────┐
│  1. Klassifikation │────▶│  2. Kategoriespezifische │
│  + Basisdaten      │     │     Detailextraktion     │
└────────────────────┘     └──────────────────────────┘
         │                              │
         ▼                              ▼
   Kundennummer, Name,            Ratenhöhe, Zieldatum,
   Geburtsdatum, etc.             Abbuchungstag, etc.
```

### Klassifikationskategorien

| Kategorie | Beschreibung |
|-----------|--------------|
| `Ratenplan anfordern` | Patient möchte Rechnungen in Raten zahlen |
| `Ratenplan unterschrieben zurücksenden` | Unterschriebener Ratenplan wird zurückgesendet |
| `Patient übermittelt Leistungsbescheid` | Leistungs-/Beihilfebescheid wird übermittelt |
| `Patient fragt erneute Zusendung des Passworts fürs Onlineportal an` | Passwort für Portal/App benötigt |
| `Patient braucht eine Rechnungskopie` | Erneute Zustellung von Rechnungen |
| `Patient möchte später zahlen` | Zahlungsaufschub gewünscht |
| `Patient teilt mit, dass er überwiesen hat` | Zahlung wurde getätigt |
| `Sonstiges` | Sonstige Anliegen |

## 📊 Ausgabeformat

Jede klassifizierte Anfrage liefert ein strukturiertes JSON:

```json
{
  "kategorie": "Ratenplan anfordern",
  "kundennummer": "1-12345-12345678",
  "vorname": "Max",
  "nachname": "Mustermann",
  "geburtsdatum": "01.01.1980",
  "anschrift": "Musterstraße 1, 12345 Berlin",
  "rechnungsbetrag": 450.0,
  "details": {
    "ratenhoehe": 50.0,
    "ratenanzahl": 9,
    "startdatum": "01.03.2025",
    "abbuchungstag": 15
  }
}
```

## 🤖 KI-Verwendung

Dieses Projekt wurde unter Verwendung von **Antigravity** (KI-Coding-Assistent) entwickelt.

### Transparenz

- **Vollständiges Code-Verständnis**: Jede Zeile wurde geprüft und verstanden
- **Eigenständige Entscheidungen**: Architektur, Modellwahl und Prompt-Design wurden bewusst gewählt

## 📁 Dateien

| Datei | Beschreibung |
|-------|--------------|
| `src/model.py` | Klassifikations- und Extraktionslogik, LLM-Integration |
| `src/evaluation.py` | Batch-Evaluation, Metriken, Visualisierungen |
| `src/prompts.py` | Prompt-Templates für das LLM |
| `data/data.csv` | Testdatensatz (83 Anfragen) |
| `output/all_predictions.json` | Vollständige Vorhersagen mit extrahierten Daten |

---

*Entwickelt mit Python, LangChain, Ollama (Llama 3), Pydantic und scikit-learn.*
