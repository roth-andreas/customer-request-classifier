import model as model_module
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap
from numpy import ndarray

# Pfade für die neue Verzeichnisstruktur
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Mapping für kürzere, lesbare Labels
LABEL_MAPPING: dict[str, str] = {
    "Ratenplan anfordern": "Ratenplan anfordern",
    "Ratenplan unterschrieben zurücksenden": "Ratenplan zurücksenden",
    "Patient übermittelt Leistungsbescheid": "Leistungsbescheid",
    "Patient fragt erneute Zusendung des Passworts fürs Onlineportal an": "Passwort anfordern",
    "Patient braucht eine Rechnungskopie": "Rechnungskopie",
    "Patient möchte später zahlen": "Später zahlen",
    "Patient teilt mit, dass er überwiesen hat": "Zahlung mitgeteilt",
    "Sonstiges": "Sonstiges"
}

def shorten_label(label: str) -> str:
    """
    Kürzt ein Label auf eine lesbare Form.

    Args:
        label: Label, der kürzer werden soll.

    Returns:
        Gekürztes Label.
    """
    return LABEL_MAPPING.get(label, label)

def plot_confusion_matrix(cm: ndarray, labels: list[str], output_path: str = "confusion_matrix.png") -> None:
    """
    Erstellt eine visuelle Heatmap der Confusion Matrix.

    Args:
        cm: Confusion Matrix.
        labels: Liste der Klassen.
        output_path: Pfad zur Speicherung des Diagramms.

    Returns:
        None
    """
    plt.figure(figsize=(14, 12))
    
    # Verwende das Label-Mapping für kürzere Labels
    short_labels = [shorten_label(l) for l in labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=short_labels, yticklabels=short_labels,
                cbar_kws={'label': 'Anzahl'})
    
    plt.xlabel('Vorhergesagt', fontsize=12)
    plt.ylabel('Tatsächlich', fontsize=12)
    plt.title('Confusion Matrix - Klassifikation der Anfragen', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix gespeichert: {output_path}")

def plot_class_distribution(y_true: list[str], y_pred: list[str], output_path: str = "class_distribution.png") -> None:
    """
    Zeigt die Verteilung der tatsächlichen vs. vorhergesagten Klassen.

    Args:
        y_true: Liste der tatsächlichen Klassen.
        y_pred: Liste der vorhergesagten Klassen.
        output_path: Pfad zur Speicherung des Diagramms.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Tatsächliche Verteilung
    true_counts = pd.Series(y_true).value_counts()
    true_counts.index = [shorten_label(l) for l in true_counts.index]
    axes[0].barh(true_counts.index, true_counts.values, color='steelblue')
    axes[0].set_title('Tatsächliche Verteilung', fontweight='bold')
    axes[0].set_xlabel('Anzahl')
    axes[0].invert_yaxis()
    
    # Vorhergesagte Verteilung
    pred_counts = pd.Series(y_pred).value_counts()
    pred_counts.index = [shorten_label(l) for l in pred_counts.index]
    axes[1].barh(pred_counts.index, pred_counts.values, color='coral')
    axes[1].set_title('Vorhergesagte Verteilung', fontweight='bold')
    axes[1].set_xlabel('Anzahl')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Klassenverteilung gespeichert: {output_path}")

def plot_metrics_per_class(report_dict: dict[str, dict[str, float]], output_path: str = "metrics_per_class.png") -> None:
    """
    Zeigt Precision, Recall, F1 pro Klasse als Balkendiagramm.

    Args:
        report_dict: Dictionary mit Metriken pro Klasse.
        output_path: Pfad zur Speicherung des Diagramms.

    Returns:
        None
    """
    # Filter nur Klassen (keine 'accuracy', 'macro avg', etc.)
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    precision = [report_dict[c]['precision'] for c in classes]
    recall = [report_dict[c]['recall'] for c in classes]
    f1 = [report_dict[c]['f1-score'] for c in classes]
    
    # Verwende das Label-Mapping für kürzere Labels
    short_classes = [shorten_label(c) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    ax.bar(x, recall, width, label='Recall', color='#3498db')
    ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_ylabel('Score')
    ax.set_title('Metriken pro Klasse', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_classes, rotation=30, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metriken pro Klasse gespeichert: {output_path}")

def evaluate() -> None:
    """
    Hauptfunktion zur Evaluation des AI-Modells.
    """
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "data.csv"), sep=";")
    y_true: list[str] = df["Anliegen"].tolist()
    
    ai_model = model_module.AIModel()
    y_pred: list[str] = []
    
    print("Starting inference on full dataset...")
    start_time = time.time()
    sample_size = len(df)
    all_predictions = []
    
    # Using tqdm for progress bar
    for idx in tqdm(range(sample_size)):
        pred = ai_model.zero_shot_classifier(f"Betreff: {df.iloc[idx]['Betreff']} \n Text: {df.iloc[idx]['Text']} \n Anlagen: {df.iloc[idx]['Anlagen']}")
        all_predictions.append(pred)
        y_pred.append(pred["kategorie"])
        
    end_time = time.time()
    print(f"Inference finished in {end_time - start_time:.2f} seconds.")
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"ACCURACY: {acc:.2%}")
    print(f"{'='*50}")
    
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\n--- Confusion Matrix ---")
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create a nice DataFrame for the CM
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)
    
    # Export CSV
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))
    
    # Save all_predictions as json
    with open(os.path.join(OUTPUT_DIR, "all_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)
    
    # ========== VISUALISIERUNGEN ==========
    print("\n--- Erstelle Visualisierungen ---")
    
    # 1. Confusion Matrix Heatmap
    plot_confusion_matrix(cm, labels, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # 2. Klassenverteilung (True vs Pred)
    plot_class_distribution(y_true, y_pred, os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # 3. Metriken pro Klasse
    plot_metrics_per_class(report, os.path.join(OUTPUT_DIR, "metrics_per_class.png"))
    
    print("\n✅ Alle Visualisierungen gespeichert!")
    print("   - confusion_matrix.png")
    print("   - class_distribution.png")
    print("   - metrics_per_class.png")

if __name__ == "__main__":
    evaluate()
