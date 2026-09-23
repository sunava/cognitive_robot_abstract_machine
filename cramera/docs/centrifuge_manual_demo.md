# Ein Handbuch ändert das Verhalten

Das Labor zeigt Dokumentwissen, beobachteten Zustand und die daraus abgeleitete
Startfreigabe gemeinsam im Bereich **Manual & robot knowledge**. Die Dokumente
beschreiben ein simuliertes CRAMERA-Demogerät und ein Demonstrationsverfahren.
Sie sind keine Herstelleranleitung für eine reale Zentrifuge.

## Kundendemo

1. Bei stillstehendem Rotor **Read manual V1** wählen. V1 verlangt mindestens zwei
   gesicherte Proben.
2. Bei geöffnetem Deckel **Load test pair** wählen und die Beladung abwarten.
   Anschließend den Deckel schließen. Zwei gleich schwere, gegenüberliegende
   Proben erfüllen die Startbedingungen.
3. **Read manual V2** wählen. Die Proben bleiben an ihren Positionen und behalten
   ihre Füllmengen. V2 verlangt vier gesicherte Proben. Der gleiche Startauftrag
   ist jetzt gesperrt; die Wissensanzeige zeigt die Änderung von zwei auf vier,
   die Dokumentstelle und die noch fehlenden Proben.
4. Den Deckel öffnen, das zweite Testpaar laden und den Deckel schließen. Der
   Start wird wieder freigegeben. **Start centrifuge** startet den simulierten Rotor.

Für den Vergleich müssen Drehzahl und Laufzeit nicht verändert werden. Ein
bereits laufender Rotor muss vor dem Einlesen eines Dokuments zum Stillstand
kommen. Das Einlesen setzt die Simulation nicht zurück. Umgekehrt bleibt das
eingelesene Handbuch bei einem physischen Reset erhalten.

## Anzeige am Gerät

Das Display auf dem Zentrifugengehäuse zeigt den beobachteten Betriebszustand,
die gemessene Drehzahl und die verbleibende Zeit als Minuten und Sekunden.
Es erhält dieselben Zustandsmeldungen wie die Bedienleiste. Das gilt unabhängig
davon, ob der Start über die Oberfläche oder einen Roboterauftrag angefordert
wird.

Der Countdown verwendet die Simulationszeit. Während des Hochfahrens bleibt
die eingestellte Laufzeit erhalten; sie zählt erst herunter, sobald die gemessene
Drehzahl innerhalb von fünf Prozent der Solldrehzahl liegt. Nach Ablauf der Zeit
zeigt das Gerät das Abbremsen bis zum tatsächlichen Stillstand. Die Anzeige
führt keinen eigenen Browser-Timer und zeigt bei unterbrochener Verbindung keine
weiterlaufende Restzeit an.

## Was tatsächlich aus dem Dokument kommt

Die Demo liest die Regel **Minimum secured samples** aus der strukturierten
Markdown-Tabelle im Abschnitt **Operating rules**. Die Dokumentansicht lässt sich
bearbeiten und erneut einlesen. Entscheidend ist der Tabelleninhalt, nicht der
Versionsname oder die Auswahl einer Schaltfläche. Die Anzeige enthält die
Dokumentversion, den Inhalts-Hash und die Quellstelle. Ungültige Dokumente werden
abgewiesen, ohne das vorherige Wissen zu verändern.

Die physische Zentrifuge prüft diese eingelesene Regel vor jedem Start – auch bei
einem direkten API-Aufruf. Ihre übrigen mechanischen Bedingungen gelten weiter:
geschlossener Deckel, ruhender Rotor, gesicherte und ausbalancierte Proben sowie
die konstruktiven Grenzen für Drehzahl und Laufzeit.

Die beobachteten Fakten stammen aus MuJoCo; sie sind keine Wahrnehmungsleistung
des PR2. Die Anzeige berechnet die aktuelle Freigabe und die fehlenden Schritte
aus diesen Fakten und der aktiven Regel. Der Verlauf hält Dokumentänderungen
fest. Dies ist eine nachvollziehbare strukturierte Dokumentauswertung, noch
kein allgemeines Sprachverständnis beliebiger Herstellerhandbücher und keine
automatische Ausführung der vorgeschlagenen Schritte durch einen CRAM-Plan.

## Schnittstellen

`POST /api/laboratory/physics/manual` und
`POST /api/laboratory/physics/robot/manual` akzeptieren entweder
`{"edition": "v1"}` / `{"edition": "v2"}` oder `{"document": "…"}`.
Die Antwort enthält den aktuellen Zustand mit `centrifuge.knowledge`, darin
`manual`, `revision`, `changes`, `history` und `evaluation`. Der geschützte
Laborzugang verwendet denselben vorhandenen Passwort- und Ursprungsschutz.

Die ausgelieferten Markdown-Dokumente liegen unter
`cramera/src/cramera/resources/centrifuge_manuals/`. Parser und Wissenszustand
liegen in `cramera/src/cramera/laboratory_manual.py`.
