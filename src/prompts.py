CLASS_PROMPT = """Du bist ein Experte für die Klassifikation von Kunden-E-Mails im medizinischen Abrechnungsbereich und für die Extraktion von hilfreichen Informationen aus den E-Mails.
            AUFGABE: Analysiere die E-Mail und wähle die EINE passendste Kategorie. 
            KATEGORIEN (wähle genau eine):
            "Ratenplan anfordern" - Patient möchte eine Ratenzahlung VEREINBAREN (z.B. "Ich möchte in Raten zahlen", "Können wir eine Ratenzahlung vereinbaren?")
            "Ratenplan unterschrieben zurücksenden" - Patient SCHICKT eine bereits unterschriebene Ratenvereinbarung ZURÜCK (z.B. "Anbei die unterschriebene Vereinbarung", "SEPA-Mandat im Anhang")
            "Patient übermittelt Leistungsbescheid" - Patient informiert über Versicherungsentscheidung oder Leistungsbescheid (z.B. "Meine Versicherung hat abgelehnt", "Leistungsbescheid anbei", "GOZ wurde nicht erstattet")
            "Patient fragt erneute Zusendung des Passworts fürs Onlineportal an" - Patient braucht Zugang zum Portal (z.B. "Ich kann mich nicht einloggen", "Passwort vergessen", "Zugang zum Portal")
            "Patient braucht eine Rechnungskopie" - Patient möchte eine Kopie/Zweitschrift der Rechnung (z.B. "Bitte senden Sie mir eine Rechnungskopie", "Zweitschrift benötigt")
            "Patient möchte später zahlen" - Patient bittet um Zahlungsaufschub OHNE Ratenzahlung (z.B. "Ich kann erst nächsten Monat zahlen", "Bitte Aufschub bis...")
            "Patient teilt mit, dass er überwiesen hat" - Patient informiert über erfolgte Zahlung (z.B. "Habe heute überwiesen", "Zahlung ist raus", "Betrag wurde überwiesen")
            "Sonstiges" - NUR wenn keine der obigen Kategorien passt
            
            EXTRAKTIONSAUFGABE: Extrahiere zusätzlich folgende Entitäten präzise aus der E-Mail und fülle die entsprechenden Felder im Output-JSON:
            "vorname" - Vorname der anfragenden Person, falls genannt.
            "nachname" - Nachname der anfragenden Person, falls genannt.
            "geburtsdatum" - Geburtsdatum der anfragenden Person, falls genannt.
            "anschrift" - Anschrift der anfragenden Person, falls genannt.
            "rechnungsbetrag" - Rechnungsbetrag der Rechnung, falls genannt.

            Wenn eine Information nicht genannt wird, setze das Feld auf null.
            ---
            E-MAIL ZU KLASSIFIZIEREN:
            {request}
            ---
            Analysiere den Inhalt. Bestimme zuerst die Kategorie. Suche danach gezielt nach den oben genannten persönlichen Informationen im Text, um alle Felder des Output-Objekts zu befüllen. Achte auf Schlüsselwörter wie "Ratenzahlung", "unterschrieben", "Leistungsbescheid", "Passwort", "Rechnungskopie", "später zahlen", "überwiesen"."""

RATENPLAN_ANFORDERUNG_PROMPT = """Extrahiere aus der folgenden E-Mail die Informationen für eine Ratenzahlungsanfrage.
            E-Mail:
            {text}
            Extrahiere:
            - ratenhoehe: Gewünschte monatliche Rate in EUR (z.B. "50 Euro" → 50.0)
            - ratenanzahl: Gewünschte Anzahl der Raten (z.B. "6 Monatsraten" → 6)
            - startdatum: Gewünschtes Startdatum (z.B. "ab 01.02.2025")
            - abbuchungstag: Tag im Monat für Abbuchung (z.B. "zum 15." → 15)
            Wenn eine Information nicht genannt wird, setze null.
            """

RECHNUNGSKOPIE_PROMPT = """Extrahiere aus der folgenden E-Mail die Informationen für eine Rechnungskopie.
            E-Mail:
            {text}
            Extrahiere:
            - anzahl_kopien: Anzahl gewünschter Kopien
            - zieladresse: Adresse für Versand (falls genannt)
            - per_email: Soll per E-Mail geschickt werden?
            Wenn eine Information nicht genannt wird, setze null.
            """

ZAHLUNGSAUFSCHUB_INFO_PROMPT = """Extrahiere aus der folgenden E-Mail die Informationen für eine Zahlungsaufschub.
            E-Mail:
            {text}
            Extrahiere:
            - zieldatum: Gewünschtes neues Zahlungsziel
            - grund: Begründung (z.B. 'warte auf Versicherung')
            Wenn eine Information nicht genannt wird, setze null.
            """