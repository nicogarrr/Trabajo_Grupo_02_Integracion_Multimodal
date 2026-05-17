param(
    [int]$WindowMs = 15000,
    [switch]$DryRun
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SafeProjectRoot = $ProjectRoot.Replace("'", "''")
$LogRoot = Join-Path $ProjectRoot "logs"
$SafeLogRoot = $LogRoot.Replace("'", "''")
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_text.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_vision.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_fusion.json") -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $ProjectRoot -Filter "event_*.tmp" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

$terminals = @(
    @{
        Title = "ChefZeroWaste - Integrador Voz"
        Log = "integrador_voz.log"
        Command = "python IntegradorMultimodal.py --window-ms $WindowMs"
    },
    @{
        Title = "ChefZeroWaste - Vision Gestos"
        Log = "vision_voz.log"
        Command = "python vision\main.py"
    },
    @{
        Title = "ChefZeroWaste - Voz ASR"
        Log = "voz_asr.log"
        Command = "python voz_asr_chef_zero_waste.py"
    }
)

Write-Host "Proyecto: $ProjectRoot"
Write-Host "Ventana temporal del integrador: $WindowMs ms"
Write-Host "Comprobando dependencias ASR..."
python -c "import sounddevice, faster_whisper, scipy, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Faltan dependencias. Ejecuta:" -ForegroundColor Red
    Write-Host "python -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}
Write-Host "Dependencias OK."

foreach ($terminal in $terminals) {
    $title = $terminal.Title.Replace("'", "''")
    $command = $terminal.Command
    $logPath = Join-Path $LogRoot $terminal.Log
    $safeLogPath = $logPath.Replace("'", "''")
    $psCommand = @"
`$Host.UI.RawUI.WindowTitle = '$title'
Set-Location -LiteralPath '$SafeProjectRoot'
Start-Transcript -Path '$safeLogPath' -Append | Out-Null
Write-Host '=== $title ==='
Write-Host 'Carpeta: $SafeProjectRoot'
Write-Host 'Comando: $command'
try {
    $command
}
finally {
    Stop-Transcript | Out-Null
}
"@

    if ($DryRun) {
        Write-Host ""
        Write-Host "[DRY RUN] $($terminal.Title)"
        Write-Host $command
    }
    else {
        Start-Process powershell.exe -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command", $psCommand
        )
        Start-Sleep -Milliseconds 700
    }
}

if (-not $DryRun) {
    Write-Host ""
    Write-Host "Ventanas lanzadas."
    Write-Host "Logs en: $SafeLogRoot"
    Write-Host "Uso: haz primero el gesto y despues habla. El integrador espera hasta $WindowMs ms."
    Write-Host "Nota: este modo usa la camara con vision\main.py y el microfono con ASR."
}
