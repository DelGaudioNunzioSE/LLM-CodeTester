@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Initial configuration ===
set LOG_FILE=test_log.log
set RESULT_FILE=test_result.txt

REM === Cleanup temporary files and folders ===
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del .coverage

REM === Run tests ===
powershell -Command "Start-Process cmd -ArgumentList '/c python -m pytest --rootdir=. --cov=solution --cov-report term > %LOG_FILE% 2>&1' -NoNewWindow -Wait"

REM === Wait a moment ===
timeout /t 2 >nul

REM === Initialize variable ===
set PASSED=0

REM === Extract number before 'passed' ===
for /f "tokens=*" %%L in ('findstr /i "passed" %LOG_FILE%') do (
    set "line=%%L"
    set "prev="
    for %%W in (!line!) do (
        if "%%W"=="passed" (
            set PASSED=!prev!
        )
        set prev=%%W
    )
)

REM === Write Pass line ===
echo Pass: !PASSED! > %RESULT_FILE%

REM === Extract and write coverage ===
for /f "tokens=4" %%a in ('findstr /C:"TOTAL" %LOG_FILE%') do (
    echo Coverage: %%a >> %RESULT_FILE%
)

exit /b 0
