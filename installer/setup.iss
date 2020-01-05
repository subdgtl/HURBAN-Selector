#define AppSetupName 'HURBAN Selector'
#define AppVersion '1.0'

[Setup]
AppName={#AppSetupName}
AppVersion={#AppVersion}
AppVerName={#AppSetupName} {#AppVersion}
AppCopyright=Copyright © 2018-2019 Subdigital
VersionInfoCompany=Subdigital
AppPublisher=Subdigital
AppPublisherURL=https://www.sub.digital/
DefaultDirName={autopf}\{#AppSetupName}
DefaultGroupName={#AppSetupName}
OutputBaseFilename={#AppSetupName}-{#AppVersion}
OutputDir=bin

PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; downloading and installing dependencies will only work if the memo/ready page is enabled (default and current behaviour)
DisableReadyPage=no
DisableReadyMemo=no

#include "scripts\lang\english.iss"

[Files]
Source: "..\target\release\hurban_selector.exe"; DestDir: "{app}"; Check: IsX64

[CustomMessages]
DependenciesDir=dependencies
WindowsServicePack=Windows %1 Service Pack %2

; shared code for installing the products
#include "scripts\products.iss"

; helper functions
#include "scripts\products\stringversion.iss"
#include "scripts\products\winversion.iss"
#include "scripts\products\fileversion.iss"
#include "scripts\products\dotnetfxversion.iss"

#include "scripts\products\msiproduct.iss"
#include "scripts\products\vsredist2015-2019.iss"


[Code]
function InitializeSetup(): boolean;
begin
	// initialize windows version
	initwinversion();

  vcredist2015_2019('14.24');

  Result := true;
end;