# Build

Build onefile funzionante verificata il 2026-04-28.

## Eseguibile unico

Obiettivo: generare un solo file `.exe`, senza cartelle o DLL da distribuire a parte.

Da PowerShell nella root del progetto:

```powershell
.\.venv\Scripts\python.exe -m nuitka main.py --onefile --onefile-no-compression --enable-plugin=pyside6 --output-dir=deployment_onefile_fast --output-filename=CenterTouch_ver3.exe --disable-cache=ccache --noinclude-qt-translations --noinclude-dlls=*.cpp.o --noinclude-dlls=*.qsb --windows-icon-from-ico=logo777.ico --include-qt-plugins=platforminputcontexts --include-data-files=logo777.ico=logo777.ico "--include-data-files=logo777_black on transparent.png=logo777_black on transparent.png" --windows-console-mode=disable
```

Output atteso:

```text
deployment_onefile_fast\CenterTouch_ver3.exe
```

Questo file puo essere copiato e avviato da solo.

## Perche questa procedura

Non usare la build solo `--standalone` se serve un unico file. Quella modalita crea una cartella tipo:

```text
deployment_nocache\main.dist\main.exe
```

In quel caso `main.exe` funziona solo se resta dentro la cartella `main.dist`, insieme a DLL, runtime Python, plugin Qt e cartelle `PySide6`, `numpy`, `shiboken6`.

Per un singolo file serve invece `--onefile`.

## Opzioni importanti

- `--onefile`: impacchetta runtime, DLL e plugin in un solo EXE.
- `--onefile-no-compression`: evita blocchi o EXE incompleti durante la compressione del payload. Il file finale e piu grande, ma piu affidabile.
- `--output-filename=CenterTouch_ver3.exe`: imposta il nome finale.
- `--include-data-files=logo777.ico=logo777.ico`: include l'icona nei dati disponibili all'app.
- `"--include-data-files=logo777_black on transparent.png=logo777_black on transparent.png"`: include il logo PNG. Le virgolette servono perche il nome file contiene spazi.
- `--windows-console-mode=disable`: evita la finestra console quando si avvia la GUI.
- `--disable-cache=ccache`: evita problemi con ccache.

## Come evitare problemi

1. Lanciare il comando da PowerShell nella root del progetto.
2. Usare la venv del progetto: `.\.venv\Scripts\python.exe`.
3. Non rinominare o spostare file asset prima della build: servono `logo777.ico` e `logo777_black on transparent.png`.
4. Se la build si blocca su permessi sotto `AppData\Local\Nuitka`, rilanciare PowerShell con permessi adeguati o eseguire il comando fuori da ambienti sandboxati.
5. Se il file onefile generato pesa poche centinaia di KB, la build non e valida: e rimasto solo lo stub. Il file corretto per questa app pesa circa 135 MB.
6. Per verificare davvero che sia onefile, copiare solo `CenterTouch_ver3.exe` in una cartella vuota e avviarlo da li.
7. Il primo avvio puo richiedere alcuni secondi: l'eseguibile estrae temporaneamente Qt e le DLL in una cartella temporanea.

## Test rapido

Dopo la build:

```powershell
New-Item -ItemType Directory -Path deployment_onefile_fast\single_test -Force
Copy-Item deployment_onefile_fast\CenterTouch_ver3.exe deployment_onefile_fast\single_test\CenterTouch_ver3.exe -Force
Start-Process -FilePath .\deployment_onefile_fast\single_test\CenterTouch_ver3.exe -WorkingDirectory .\deployment_onefile_fast\single_test
```

Se l'app si apre anche da `single_test`, il file e realmente distribuibile da solo.
