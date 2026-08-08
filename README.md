# Interview Assistant

Applicazione desktop per Windows che affianca il selezionatore durante
un colloquio di lavoro: ascolta la conversazione, la trascrive dal vivo
distinguendo chi parla, e al termine genera un report strutturato del
candidato. Tutto in locale, senza servizi a pagamento.

## Cosa fa

Durante il colloquio l'applicazione sta accanto alla finestra della
videochiamata e registra due sorgenti audio separate: il microfono (la
voce del selezionatore) e l'audio in uscita dal computer (la voce del
candidato in videochiamata). Tenerle separate è ciò che permette di
attribuire ogni frase alla persona giusta, come in un dialogo scritto.

Poiché legge l'audio in uscita dal computer, funziona con **Microsoft
Teams, Zoom, Google Meet, Webex e qualsiasi altra piattaforma**, senza
installare nulla dentro la videochiamata e senza chiedere permessi
all'organizzatore.

Al termine, la trascrizione viene passata a un modello linguistico che
gira sul computer stesso e produce un report con sintesi, punti di
forza, aree di attenzione, competenze citate, domande di follow-up e
una valutazione complessiva. Il colloquio viene salvato in un archivio
locale ed è esportabile in Word o testo.

## Perché è gratuito per sempre

Non ci sono chiavi API né abbonamenti: entrambi i modelli girano sul
computer del cliente e hanno licenze che ne consentono l'uso
commerciale senza royalty.

- **Whisper** (motore `faster-whisper`) per la trascrizione — licenza MIT.
  Riconosce automaticamente circa 99 lingue: il colloquio può svolgersi
  in italiano, inglese o qualunque altra lingua senza configurare nulla.
- **Qwen2.5-3B-Instruct** per i report — licenza Apache 2.0, multilingue.

I modelli (circa 2,5 GB in totale) vengono scaricati una sola volta al
primo avvio. Dopodiché l'applicazione funziona completamente offline e
nessun dato lascia il computer.

## Come si ottiene il file di installazione

Il progetto include una procedura di compilazione automatica: a ogni
modifica del codice, GitHub compila l'applicazione su una macchina
Windows e produce `InterviewAssistantSetup.exe`, il file unico da
consegnare al cliente.

Per scaricarlo: scheda **Actions** del repository → apri l'ultima
esecuzione completata con il segno verde → sezione **Artifacts** in
fondo alla pagina → `InterviewAssistantSetup`.

La compilazione include una verifica automatica del pacchetto: se una
libreria necessaria non finisse nell'eseguibile, la compilazione
fallisce subito invece di produrre un installer che si romperebbe solo
sul computer del cliente.

In alternativa si può compilare in locale su un PC Windows con Python
3.11 a 64 bit, eseguendo `build\build.bat` dalla cartella principale e
poi aprendo `build\installer.iss` con [Inno Setup](https://jrsoftware.org/isinfo.php)
(gratuito).

## Requisiti del computer del cliente

- Windows 10 o 11 a 64 bit
- Processore Intel o AMD recente (vedi la nota sui processori ARM64)
- Circa 6 GB liberi su disco
- 8 GB di RAM consigliati
- Connessione a internet solo al primo avvio

### Nota sui processori ARM64

Sui PC Windows con processore ARM64 (per esempio i portatili
Snapdragon) l'applicazione gira in emulazione. Le librerie di calcolo
usano istruzioni che l'emulazione non sempre supporta: in quel caso il
programma verrebbe chiuso dal sistema operativo senza alcun messaggio.

Per questo, al primo avvio, l'applicazione esegue una verifica in un
processo separato e, se necessario, attiva automaticamente una
**modalità compatibilità** più lenta ma stabile. La modalità si può
anche forzare a mano dalla scheda Impostazioni.

Su un normale PC Intel o AMD questa verifica passa al primo tentativo e
le prestazioni sono sensibilmente migliori.

## Struttura del progetto

```
interview-assistant/
├── app/
│   ├── main.py               avvio dell'interfaccia
│   ├── config.py             percorsi e costanti
│   ├── settings.py           preferenze dell'utente
│   ├── compat.py             rilevamento CPU e modalità compatibilità
│   ├── diagnostics.py        verifica di avvio in processo separato
│   ├── logging_setup.py      registro eventi e cattura dei crash
│   ├── platform_detect.py    rilevamento Teams/Zoom/Meet
│   ├── audio/capture.py      microfono + audio di sistema
│   ├── transcription/        trascrizione dal vivo con Whisper
│   ├── summarization/        generazione del report
│   ├── storage/db.py         archivio colloqui (SQLite)
│   ├── export/report.py      esportazione Word e testo
│   └── ui/                   interfaccia grafica (PySide6/Qt)
├── build/
│   ├── app.spec              configurazione PyInstaller
│   ├── build.bat             compilazione locale
│   └── installer.iss         creazione dell'installer
├── .github/workflows/        compilazione automatica su GitHub
├── requirements.txt
└── run.py                    punto di ingresso
```

## Diagnostica

Se qualcosa non funziona sul computer del cliente, la scheda
**Impostazioni** contiene il pulsante *Apri la cartella dei log*. La
cartella si trova in `%APPDATA%\InterviewAssistant\logs` e contiene:

- `interview-assistant-AAAA-MM-GG.log` — cronologia dettagliata;
- `ultimo-crash-principale.txt` — punto esatto di un'eventuale chiusura
  improvvisa dell'applicazione;
- `ultimo-crash-test-motore.txt` — stesso dato per la verifica iniziale.

## Cosa manca ancora

Questo è un MVP funzionante, non un prodotto finito. Prima di una
distribuzione ampia andrebbero considerati:

- **Firma digitale del codice** (circa 100-300 € l'anno) per evitare
  l'avviso "Windows ha protetto il tuo PC" al primo avvio.
- **Informativa privacy**: registrare un colloquio ha implicazioni GDPR.
  È opportuno che il cliente informi i candidati e raccolga il consenso;
  un avviso dentro l'applicazione sarebbe un buon aggiunta.
- **Icona e nome definitivi** del cliente al posto di quelli attuali.
- **Prova sul campo**: la cattura dell'audio di sistema dipende dai
  driver audio del PC e va verificata su qualche configurazione reale.
- **Suggerimenti in tempo reale** durante il colloquio (ora l'AI
  interviene solo alla fine) e **accelerazione GPU** per i PC dotati di
  scheda video dedicata.
