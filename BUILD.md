# Build

Build funzionante verificata il 2026-04-23.

## Eseguibile standalone

Da PowerShell nella root del progetto:

```powershell
.\.venv\Scripts\python.exe -m nuitka main.py --enable-plugin=pyside6 --output-dir=deployment_nocache --disable-cache=ccache --noinclude-qt-translations --standalone --noinclude-dlls=*.cpp.o --noinclude-dlls=*.qsb --windows-icon-from-ico=logo777.ico --include-qt-plugins=platforminputcontexts --windows-console-mode=disable
```

Output atteso:

```text
deployment_nocache\main.dist\main.exe
```

## Nota

`pyside6-deploy` in questo ambiente puo bloccarsi per permessi/cache Nuitka sotto `AppData\Local\Nuitka`.
La build diretta con `nuitka`, `--disable-cache=ccache` e `--windows-console-mode=disable` e risultata affidabile.
