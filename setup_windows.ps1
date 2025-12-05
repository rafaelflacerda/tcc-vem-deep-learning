Write-Host '[1/3] Creating virtual environment (.venv)...'
python -m venv .venv

if (!(Test-Path '.venv')) {
    Write-Host 'Error: The virtual environment could not be created'
    exit 1
}

Write-Host '[2/3] Activating environment...'
& .\.venv\Scripts\Activate.ps1

Write-Host '[3/3] Installing requirements...'
pip install --upgrade pip
pip install -r requirements.txt

Write-Host 'Done! Environment is active.'