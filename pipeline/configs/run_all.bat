@echo off
setlocal enabledelayedexpansion

REM Cartella radice (modifica se vuoi)
set "ROOT_DIR=%cd%"

REM Conta le sottocartelle immediate (primo livello)
set count=0
for /d %%D in ("%ROOT_DIR%\*") do (
    set /a count+=1
)

echo Trovate %count% sottocartelle da testare.

set i=0

REM Cicla solo sulle sottocartelle immediate
for /d %%D in ("%ROOT_DIR%\*") do (
    set /a i+=1
    set "run_bat=%%D\run_test.bat"

    if exist "!run_bat!" (
        echo.
        echo ================================
        echo Eseguo test !i! di %count% in: %%D
        echo ================================
        pushd "%%D"
        call run_test.bat
        popd
    ) else (
        echo.
        echo Sottocartella %%D non contiene run_test.bat, salto.
    )
)

echo.
echo Tutti i test sono stati eseguiti.
pause
