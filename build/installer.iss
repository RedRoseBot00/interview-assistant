; ============================================================
; Script Inno Setup per creare l'installer .exe di
; AI Interview Assistant.
;
; Prerequisiti:
;   1. Aver gia' eseguito build\build.bat (che genera dist\InterviewAssistant\)
;   2. Aver installato Inno Setup (https://jrsoftware.org/isinfo.php) - gratuito
;
; Uso:
;   Apri questo file con Inno Setup Compiler e premi "Compile".
;   Il risultato sara' Output\InterviewAssistantSetup.exe, pronto per
;   essere consegnato al cliente: un doppio clic installa l'app senza
;   bisogno di Python o di conoscenze tecniche.
; ============================================================

#define MyAppName "AI Interview Assistant"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Il tuo nome / la tua azienda"
#define MyAppExeName "InterviewAssistant.exe"

[Setup]
AppId={{8F2C0B5A-6B7E-4E6C-9B7B-INTERVIEWAI01}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=InterviewAssistantSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul desktop"; GroupDescription: "Icone aggiuntive:"

[Files]
Source: "..\dist\InterviewAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Avvia {#MyAppName}"; Flags: nowait postinstall skipifsilent
