; ============================================================================
; SDR Module Inno Setup Installer Script
; ============================================================================
; This script creates a Windows installer for the SDR Module application.
;
; Prerequisites:
;   - Inno Setup 6.x (https://jrsoftware.org/isinfo.php)
;   - Built executable in dist\sdr-module\ directory
;
; Usage:
;   1. Build the executable first: build_windows.bat
;   2. Open this file in Inno Setup Compiler
;   3. Click Build -> Compile
;
; Or from command line:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
; ============================================================================

#define MyAppName "SDR Module"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "SDR Module Team"
#define MyAppURL "https://github.com/sdr-module"
#define MyAppExeName "sdr-scan.exe"

[Setup]
; Application identification
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation settings
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=installer_output
OutputBaseFilename=SDR-Module-{#MyAppVersion}-Setup
SetupIconFile=
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

; Minimum Windows version (Windows 10)
MinVersion=10.0

; Privileges required
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Uninstaller settings
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "Add to system PATH"; GroupDescription: "System Integration:"; Flags: unchecked

[Files]
; Main application files from PyInstaller output
Source: "dist\sdr-module\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "SPEC_SHEET.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu icons
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} Help"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--help"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{group}\Documentation"; Filename: "{app}\SPEC_SHEET.md"

; Desktop icon (optional)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Add to PATH if selected
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath('{app}')

[Run]
; Post-install actions
Filename: "{app}\{#MyAppExeName}"; Parameters: "--help"; Description: "View command-line help"; Flags: nowait postinstall skipifsilent shellexec

[Code]
// Check if path needs to be added
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

// Remove from PATH on uninstall
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path: string;
  AppPath: string;
  P: Integer;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    if RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path) then
    begin
      AppPath := ExpandConstant('{app}');
      P := Pos(';' + AppPath, Path);
      if P > 0 then
      begin
        Delete(Path, P, Length(';' + AppPath));
        RegWriteStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
      end
      else
      begin
        P := Pos(AppPath + ';', Path);
        if P > 0 then
        begin
          Delete(Path, P, Length(AppPath + ';'));
          RegWriteStringValue(HKLM, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', Path);
        end;
      end;
    end;
  end;
end;

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nSDR Module is a software-defined radio framework for signal processing, demodulation, and analysis.%n%nIt is recommended that you close all other applications before continuing.
