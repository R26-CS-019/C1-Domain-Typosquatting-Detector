$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"

$Directories = @(
    "data\processed",
    "data\splits",
    "models",
    "reports",
    "results",
    "scripts",
    "src",
    "tests"
)

foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

if (-not (Test-Path "ai_research_env\Scripts\python.exe")) {
    Write-Host "Creating Python 3.12 virtual environment..."
    py -3.12 -m venv ai_research_env
}

$Python = Join-Path $ProjectRoot "ai_research_env\Scripts\python.exe"

Write-Host "Installing project dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup completed. Next command:"
Write-Host '.\ai_research_env\Scripts\python.exe -X utf8 -u .\scripts\01_audit_phiusiil.py'

