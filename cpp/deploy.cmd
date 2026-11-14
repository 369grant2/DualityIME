@echo off
rem Copies a fresh build into cpp\build\stage (where register.cmd registered it).
rem A DLL still loaded in some app is renamed out of the way first: new apps get the new one.
setlocal
set STAGE=%~dp0build\stage
set B=%~dp0build
taskkill /im DualityIMEEngine.exe /f >nul 2>&1
for %%f in (DualityIME64.dll DualityIME32.dll) do (
    if exist "%STAGE%\%%f" (
        del /q "%STAGE%\%%f.old*" >nul 2>&1
        copy /y nul "%STAGE%\%%f.probe" >nul & del /q "%STAGE%\%%f.probe"
        move /y "%STAGE%\%%f" "%STAGE%\%%f.old%RANDOM%" >nul
    )
)
copy /y "%B%\x64-Release\DualityIME64.dll" "%STAGE%\" >nul || exit /b 1
copy /y "%B%\x86-Release\DualityIME32.dll" "%STAGE%\" >nul || exit /b 1
copy /y "%B%\x64-Release\DualityIMEEngine.exe" "%STAGE%\" >nul || exit /b 1
rem The engine reads data\ next to itself: the repository's, linked (a junction: no copy of the 50 MB).
rem Without it the engine the DLL starts finds no dictionary and typing hangs (2026-11-13).
rem A link to a folder that is gone (the repository moved) still "exists": judged by the dictionary in it,
rem and only the link is removed (rmdir without /s).
for %%d in ("%~dp0..\data") do set DATA=%%~fd
if exist "%STAGE%\data" if not exist "%STAGE%\data\lexicon.txt" rmdir "%STAGE%\data"
if not exist "%STAGE%\data" mklink /j "%STAGE%\data" "%DATA%" >nul || exit /b 1
echo deployed
