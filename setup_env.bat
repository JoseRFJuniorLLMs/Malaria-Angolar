@echo off
set "VENV_NAME=venv"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "%VENV_NAME%" (
    echo Creating virtual environment '%VENV_NAME%'...
    python -m venv %VENV_NAME%
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment '%VENV_NAME%' already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
if exist "requirements.txt" (
    echo Installing requirements from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found in current directory.
    echo Installing base requirements for YOLOv8...
    pip install ultralytics opencv-python matplotlib tqdm
)

echo.
echo Environment setup complete!
echo To activate the environment manually, run: %VENV_NAME%\Scripts\activate
echo.
pause
