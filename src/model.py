from typing import Literal, Optional, Any
import re
import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from prompts import CLASS_PROMPT, RATENPLAN_ANFORDERUNG_PROMPT, RECHNUNGSKOPIE_PROMPT, ZAHLUNGSAUFSCHUB_INFO_PROMPT

class AIModel:
    class ClassificationResponse(BaseModel):
        category: Literal["Ratenplan anfordern", "Ratenplan unterschrieben zurücksenden", "Patient übermittelt Leistungsbescheid",
         "Patient fragt erneute Zusendung des Passworts fürs Onlineportal an", "Patient braucht eine Rechnungskopie",
          "Patient möchte später zahlen", "Patient teilt mit, dass er überwiesen hat", "Sonstiges"] = Field(description="The most probable category of the request")
        vorname: Optional[str] = Field(description="The surname of the patient if available, otherwise empty string")
        nachname: Optional[str] = Field(description="The name of the patient if available, otherwise empty string")
        rechnungsbetrag: Optional[float] = Field(description="The amount of the invoice if available, otherwise 0.0")
        geburtsdatum: Optional[str] = Field(description="The date of birth of the patient if available, otherwise empty string")
        anschrift: Optional[str] = Field(description="The address of the patient if available, otherwise empty string")

    class RatenplanAnforderung(BaseModel):
        """Extrahierte Infos bei 'Ratenplan anfordern'"""
        ratenhoehe: Optional[float] = Field(default=0.0, description="Gewünschte Ratenhöhe in EUR")
        ratenanzahl: Optional[int] = Field(default=0, description="Gewünschte Anzahl der Raten")
        startdatum: Optional[str] = Field(default="", description="Gewünschtes Startdatum (z.B. '01.02.2025')")
        abbuchungstag: Optional[int] = Field(default=0, description="Tag im Monat für Abbuchung (1-28)") 

    class Rechnungskopie(BaseModel):
        """Extrahierte Infos bei 'Patient braucht Rechnungskopie'"""
        anzahl_kopien: Optional[int] = Field(default=1, description="Anzahl gewünschter Kopien")
        zieladresse: Optional[str] = Field(default="", description="Adresse für Versand (falls genannt)")
        per_email: Optional[bool] = Field(default=False, description="Soll per E-Mail geschickt werden?")
    class Zahlungsaufschub(BaseModel):
        """Extrahierte Infos bei 'Patient möchte später zahlen'"""
        zieldatum: Optional[str] = Field(default="", description="Gewünschtes neues Zahlungsziel")
        grund: Optional[str] = Field(default="", description="Begründung (z.B. 'warte auf Versicherung')")           

    def __init__(self) -> None:
        """
        Initialisiert den AI-Modell.

        Args:
            None

        Returns:
            None
        """
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        labels_path = os.path.join(base_dir, "data", "classification_targets.txt")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]
        
        # Ollama Host: Lokal oder Docker
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.llm = ChatOllama(model="llama3", temperature=0, base_url=ollama_host)

    def extract_ratenplan_info(self, text: str) -> Optional[dict[str, Any]]:
        """
        Extrahiert Informationen aus der Mail, die einen Ratenplan anfordert.

        Args:
            text: Text der Mail, die analysiert werden soll.

        Returns:
            Optional[dict[str, Any]]: Extrahierte Informationen als Dictionary.
        """
        try:
            prompt = ChatPromptTemplate.from_template(RATENPLAN_ANFORDERUNG_PROMPT)
            return self.llm.with_structured_output(self.RatenplanAnforderung).invoke(prompt.format(text=text)).model_dump(mode='json')
        except Exception as e:
            print(f"Error in extract_ratenplan_info: {e}")
            return None

    def extract_rechnungskopie_info(self, text: str) -> Optional[dict[str, Any]]:
        """
        Extrahiert Informationen aus der Mail, die eine Rechnungskopie anfordert.

        Args:
            text: Text der Mail, die analysiert werden soll.

        Returns:
            Optional[dict[str, Any]]: Extrahierte Informationen als Dictionary.
        """
        try:
            prompt = ChatPromptTemplate.from_template(RECHNUNGSKOPIE_PROMPT)
            return self.llm.with_structured_output(self.Rechnungskopie).invoke(prompt.format(text=text)).model_dump(mode='json')
        except Exception as e:
            print(f"Error in extract_rechnungskopie_info: {e}")
            return None

    def extract_zahlungsaufschub_info(self, text: str) -> Optional[dict[str, Any]]:
        """
        Extrahiert Informationen aus der Mail, die eine Zahlungsaufschub anfordert.

        Args:
            text: Text der Mail, die analysiert werden soll.

        Returns:
            Optional[dict[str, Any]]: Extrahierte Informationen als Dictionary.
        """
        try:
            prompt = ChatPromptTemplate.from_template(ZAHLUNGSAUFSCHUB_INFO_PROMPT)
            return self.llm.with_structured_output(self.Zahlungsaufschub).invoke(prompt.format(text=text)).model_dump(mode='json')
        except Exception as e:
            print(f"Error in extract_zahlungsaufschub_info: {e}")
            return None

    def step2_classifier(self, request: str, predictions: ClassificationResponse) -> Optional[dict[str, Any]]:
        """
        Extrahiert weitergehende Informationen aus der Mail.

        Args:
            request: Text der Mail, die analysiert werden soll.
            predictions: Ergebnisse der Klassifikation.

        Returns:
            Optional[dict[str, Any]]: Extrahierte Informationen als Dictionary.
        """
        extracted_info = None
        if predictions.category == "Ratenplan anfordern":
            extracted_info = self.extract_ratenplan_info(request)
        elif predictions.category == "Patient braucht eine Rechnungskopie":
            extracted_info = self.extract_rechnungskopie_info(request)
        elif predictions.category == "Patient möchte später zahlen":
            extracted_info = self.extract_zahlungsaufschub_info(request)
    
        return extracted_info


    def zero_shot_classifier(self, request: str) -> dict[str, Any]:
        """
        Klassifiziert die Mail mittels lokalem Ollama (Llama 3).

        Args:
            request: Text der Mail, die analysiert werden soll.

        Returns:
            dict[str, Any]: Ergebnisse der Klassifikation.
        """
        prompt_template = CLASS_PROMPT
        prompt = ChatPromptTemplate.from_template(prompt_template)
        try:
            res = self.llm.with_structured_output(self.ClassificationResponse).invoke(prompt.format(request=request, categories=json.dumps(self.labels, ensure_ascii=False)))
            step2_results = self.step2_classifier(request, res)
            return {"kategorie": res.category, "vorname": res.vorname, "nachname": res.nachname,
             "rechnungsbetrag": res.rechnungsbetrag, "geburtsdatum": res.geburtsdatum, "anschrift": res.anschrift, "kundennummer": extract_personal_information(request), "details": step2_results}
        except Exception as e:
            print(f"Error in zero_shot_classifier: {e}")
            return {"kategorie": "Sonstiges", "vorname": "", "nachname": "",
             "rechnungsbetrag": 0.0, "geburtsdatum": "", "anschrift": "", "kundennummer": extract_personal_information(request), "details": None, "error": str(e)}

def extract_personal_information(request: str) -> str:
    """
    Extrahiert Kundennummern aus der Anfrage.
    Beispiele: 4-95181-83140807, 8-87236-47338124

    Args:
        request: Text der Mail, die analysiert werden soll.

    Returns:
        str: Kundennummer.
    """
    # Format: X-XXXXX-XXXXXXXX
    regex = r"\d-\d{5}-\d{8}(?!\d)"
    matches = re.findall(regex, request)
    # Es gibt keine Fälle mit unterschiedlichen Kundennummern im Datensatz
    return str(np.unique(matches)[0]) if len(np.unique(matches)) > 0 else ""