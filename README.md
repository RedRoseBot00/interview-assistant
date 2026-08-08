# AI Interview Assistant — MVP

Assistente desktop per Windows che aiuta il recruiter a condurre colloqui
di lavoro: registra e trascrive il colloquio in tempo reale (in qualsiasi
lingua) e, al termine, genera automaticamente un report strutturato
(sintesi, punti di forza, aree di attenzione, domande di follow-up,
valutazione) usando un modello AI **locale**, senza inviare nulla al
cloud e senza costi per singola chiamata.

## Come funziona (in breve)

1. Il recruiter apre l'app, inserisce nome candidato e posizione, preme
   "Inizia colloquio".
2. L'app registra sia il microfono che l'audio di sistema (es. la voce
   del candidato in una videochiamata Teams/Zoom) e la trascrive in
   tempo reale con **Whisper** (motore `faster-whisper`, 100% locale,
   riconoscimento automatico della lingua — quindi supporta italiano,
   inglese e qualunque altra lingua senza configurazione).
3. Alla fine del colloquio, il recruiter preme "Termina e genera
   report": la trascrizione viene passata a un **LLM locale**
   (Qwen2.5-3B-Instruct, licenza Apache 2.0, gratuito per sempre, gira
   sulla CPU del PC senza bisogno di GPU né di connessione a servizi a
   pagamento) che genera il report.
4. Il colloquio (trascrizione + report) viene salvato in un archivio
   locale (SQLite) ed è esportabile in Word (.docx) o testo (.txt).

Nessun dato lascia il PC del cliente: non ci sono chiavi API, non ci
sono abbonamenti, non ci sono costi ricorrenti.

## ⚠️ Limite importante di questa consegna

Questo progetto è stato scritto in un ambiente cloud Linux, che **non
può compilare un .exe Windows funzionante** (mancano le librerie audio
Windows/WASAPI, i binari nativi di llama.cpp per Windows, e non è
possibile testare cattura audio o interfaccia grafica senza hardware
reale). Quello che ricevi è il **codice sorgente completo e funzionante
concettualmente**, pronto per essere compilato in un vero `.exe` su una
macchina Windows.

**Non serve però un tuo PC Windows**: più sotto trovi un'opzione per
compilare il .exe automaticamente e gratuitamente su un server Windows
messo a disposizione da GitHub, senza installare nulla sul tuo computer.

La build vera e propria (10-15 minuti, procedura guidata sotto) va fatta
una volta, da te (lo sviluppatore), su un PC Windows. Il tuo cliente
riceverà solo il file `InterviewAssistantSetup.exe` finale — lui non
deve installare Python né toccare il codice.

## Struttura del progetto

```
interview-assistant/
├── app/
│   ├── main.py                  punto di ingresso dell'app
│   ├── config.py                percorsi, nomi modelli, impostazioni
│   ├── audio/capture.py         cattura microfono + audio di sistema
│   ├── transcription/engine.py  trascrizione live con Whisper locale
│   ├── summarization/llm.py     generazione report con LLM locale
│   ├── storage/db.py            archivio colloqui (SQLite)
│   ├── export/report.py         esportazione Word/testo
│   ├── models/download.py       download automatico modello AI al 1° avvio
│   └── ui/                      interfaccia grafica (PySide6/Qt)
├── build/
│   ├── app.spec                 configurazione PyInstaller
│   ├── build.bat                script di build automatico
│   └── installer.iss            script Inno Setup per il .exe finale
├── requirements.txt
└── run.py                       entry point per sviluppo e per PyInstaller
```

## Opzione A (consigliata): build automatica gratuita, senza PC Windows

Il progetto include già `.github/workflows/build-windows.yml`: una
"ricetta" che fa compilare l'app a un PC Windows temporaneo messo a
disposizione gratuitamente da GitHub. Tu non tocchi codice, non installi
Python, non hai bisogno di Windows — solo un account GitHub gratuito e
un browser.

1. Vai su [github.com](https://github.com) e crea un account gratuito
   (se non ne hai già uno).
2. Crea un nuovo repository (pulsante verde "New"), dagli un nome (es.
   `interview-assistant`), lascialo **pubblico** (così le build sono
   illimitate e gratuite) e crealo senza aggiungere altri file.
3. Nella pagina del repository appena creato, usa il link "uploading an
   existing file" (o trascina i file) per caricare **tutto il contenuto**
   di questa cartella `interview-assistant` (inclusa la sottocartella
   nascosta `.github`, importante: se il tuo browser non la mostra,
   trascina l'intera cartella invece dei singoli file). Conferma il
   commit.
4. Vai sulla scheda **Actions** del repository: dovresti vedere il
   workflow "Build Windows installer". Cliccaci sopra, poi premi il
   pulsante **"Run workflow"** (in alto a destra) e conferma.
5. Attendi 10-15 minuti mentre GitHub compila l'app su una macchina
   Windows vera. Quando la build è verde (✓), apri la build completata
   e in fondo alla pagina trovi la sezione **"Artifacts"**: scarica
   `InterviewAssistantSetup` — è uno zip che contiene il file
   `InterviewAssistantSetup.exe` **già pronto, vero e testabile**, da
   consegnare al tuo cliente.

Questo file .exe è generato davvero su Windows (non è una simulazione),
quindi è quello definitivo. Se una build fallisce, la scheda Actions
mostra il log dettagliato dell'errore — in tal caso incollamelo e ti
aiuto a correggere il codice.

## Opzione B: build manuale sul tuo PC Windows

Se preferisci compilare in locale (ad esempio per testare subito
microfono e audio di sistema sul tuo PC prima di consegnare l'app),
puoi farlo così:

### Prerequisiti (una tantum, sul tuo PC di sviluppo)

1. **Python 3.10 o 3.11 (64 bit)** — [python.org/downloads](https://www.python.org/downloads/)
   durante l'installazione spunta "Add python.exe to PATH".
2. **Inno Setup** (gratuito) — [jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php)
   serve solo per l'ultimo passaggio (creare l'installer).
3. Circa **6 GB liberi** su disco per build e modelli.

### Passaggi

1. Copia questa cartella `interview-assistant` sul PC Windows.
2. Apri il Prompt dei comandi nella cartella del progetto ed esegui:

   ```
   build\build.bat
   ```

   Lo script crea un ambiente virtuale Python, installa tutte le
   dipendenze e compila l'app con PyInstaller. Al termine troverai
   l'app funzionante in `dist\InterviewAssistant\InterviewAssistant.exe`
   — provala subito per verificare che tutto funzioni sul tuo PC
   (microfono, audio di sistema, generazione report).

3. Apri `build\installer.iss` con Inno Setup Compiler e premi **Compile**
   (o F9). Otterrai `build\Output\InterviewAssistantSetup.exe`: **questo
   è il file da consegnare al tuo cliente**. Un doppio clic lo installa
   come qualunque altro programma Windows, con icona sul desktop e nel
   menu Start.

4. Al primo avvio, l'app scarica automaticamente il modello AI locale
   (~2 GB, una tantum, richiede internet). Le volte successive parte
   subito, offline.

### Nota su Microsoft Defender SmartScreen

Poiché l'eseguibile non è firmato digitalmente, al primo avvio Windows
potrebbe mostrare un avviso "Windows ha protetto il tuo PC". È normale
per applicazioni non firmate: basta cliccare "Ulteriori informazioni" →
"Esegui comunque". Per evitare questo avviso in modo definitivo serve
un certificato di firma del codice (a pagamento, ~100-300 €/anno) — non
necessario per un MVP, ma da considerare se l'app verrà distribuita più
ampiamente.

## Licenze dei modelli usati (tutte gratuite per uso commerciale)

- **Whisper / faster-whisper**: licenza MIT.
- **Qwen2.5-3B-Instruct**: licenza Apache 2.0.

Entrambi possono essere usati liberamente, anche commercialmente, senza
royalty e senza scadenza: questo è ciò che rende l'app "gratuita per
sempre" come richiesto — non ci sono chiamate API a pagamento, tutto
gira in locale sul PC del cliente.

## Limiti dell'MVP attuale / cosa manca

Questo è un MVP funzionale ma minimale. Cose da rifinire prima di un
uso in produzione con un vero cliente:

- **Test su hardware reale**: cattura audio, prestazioni del modello e
  qualità della trascrizione vanno verificate su PC Windows reali
  (specialmente la cattura dell'audio di sistema, che dipende dai
  driver audio del PC).
- **Prestazioni**: su PC senza GPU dedicata, sia Whisper che l'LLM
  girano su CPU. Con un portatile di fascia media la trascrizione live
  dovrebbe reggere il passo del parlato, ma su PC molto datati potrebbe
  accumulare ritardo. Si può mitigare scegliendo un modello Whisper più
  piccolo (`base` invece di `small`) nelle impostazioni.
- **Gestione permessi/privacy**: registrare un colloquio con audio del
  candidato ha implicazioni di privacy (GDPR) — è consigliabile che il
  tuo cliente informi e ottenga il consenso dei candidati alla
  registrazione, ed è buona norma aggiungere un banner/avviso in-app in
  tal senso.
- **Firma del codice** per evitare gli avvisi SmartScreen (vedi sopra).
- **Icona applicazione**: `app/resources/icon.ico` è un'icona
  segnaposto generata automaticamente — sostituiscila con il logo del
  cliente prima della consegna finale.
- **UI/UX**: l'interfaccia è funzionale ma essenziale; si può curare
  ulteriormente (branding, temi, indicatori di livello audio, ecc.) in
  una fase successiva.

## Possibili evoluzioni future (fuori dallo scope dell'MVP)

- Suggerimenti AI in tempo reale durante il colloquio (non solo il
  report finale).
- Overlay sempre-in-primo-piano compatibile con le videochiamate.
- Accelerazione GPU opzionale per modelli più potenti (LLM 7-8B) su PC
  con scheda video dedicata.
- Sincronizzazione/backup opzionale dello storico colloqui (rimanendo
  comunque locale/offline se richiesto).
