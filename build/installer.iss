; ============================================================
; Script Inno Setup per creare l'installer di Interview Assistant.
;
; Prerequisiti:
;   1. aver gia' eseguito build\build.bat (che genera dist\InterviewAssistant\)
;   2. aver installato Inno Setup (gratuito): https://jrsoftware.org/isinfo.php
;
; Uso:
;   apri questo file con Inno Setup Compiler e premi Compile.
;   Il risultato e' Output\InterviewAssistantSetup.exe, il file unico da
;   consegnare al cliente: un doppio clic lo installa come qualunque
;   altro programma Windows, senza bisogno di Python.
; ============================================================

#define MyAppName "Interview Assistant"
#define MyAppVersion "0.2.1"
#define MyAppPublisher "Interview Assistant"
#define MyAppExeName "InterviewAssistant.exe"

[Setup]
; Identificativo univoco del programma: NON va piu' cambiato, altrimenti
; gli aggiornamenti verrebbero installati accanto alla versione vecchia
; invece di sostituirla.
AppId={{7E1D3F04-2A5B-4C6D-9E8F-0B1A2C3D4E5F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}
DefaultDirName={autopf}\InterviewAssistant
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=InterviewAssistantSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

; "x64compatible" comprende sia i PC con processore Intel/AMD a 64 bit
; sia quelli con processore ARM64 che eseguono programmi x64 in
; emulazione. L'identificatore "x64", usato spesso per abitudine,
; escluderebbe proprio questi ultimi.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Se il programma e' aperto durante un aggiornamento, Inno Setup chiede
; di chiuderlo invece di lasciare file bloccati e installazione a meta'.
CloseApplications=yes
RestartApplications=no
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

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

[Code]
// I modelli AI scaricati occupano alcuni gigabyte e restano fuori dalla
// cartella di installazione: senza questa domanda resterebbero sul
// disco anche dopo la disinstallazione, senza che l'utente lo sappia.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if MsgBox('Vuoi eliminare anche i modelli AI scaricati e i colloqui salvati?' + #13#10 +
              'Occupano alcuni gigabyte in %APPDATA%\InterviewAssistant.' + #13#10 + #13#10 +
              'Scegli No se intendi reinstallare il programma piu'' avanti.',
              mbConfirmation, MB_YESNO) = IDYES then
      DelTree(ExpandConstant('{userappdata}\InterviewAssistant'), True, True, True);
  end;
end;
