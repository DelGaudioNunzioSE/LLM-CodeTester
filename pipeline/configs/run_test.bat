@echo off
REM === Initial configuration ===
set LOG_FILE=test_log.log
set RESULT_FILE=test_result.txt

REM === Cleanup temporary files and folders ===
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del .coverage

REM === Start tests using PowerShell ===
powershell -Command "Start-Process cmd -ArgumentList '/c python -m pytest --rootdir=. --cov=solution --cov-report term > %LOG_FILE% 2>&1' -NoNewWindow -Wait"

REM === Wait for log file to be written ===
timeout /t 2 >nul

REM === Check test result from log ===
findstr /C:"= FAILURES =" %LOG_FILE% >nul
if %errorlevel%==0 (
    echo Some tests failed. Check the output for details.
    echo Fail > %RESULT_FILE%
    set EXIT_CODE=1
) else (
    echo Tests completed successfully.
    echo Pass > %RESULT_FILE%
    set EXIT_CODE=0
)

REM === Extract coverage from the log ===
for /f "tokens=4" %%a in ('findstr /C:"TOTAL" %LOG_FILE%') do (
    echo Coverage: %%a >> %RESULT_FILE%
)

exit /b %EXIT_CODE%
