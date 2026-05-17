param(
    [int]$WindowMs = 8000,
    [switch]$DryRun
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SafeProjectRoot = $ProjectRoot.Replace("'", "''")
$LogRoot = Join-Path $ProjectRoot "logs"
$SafeLogRoot = $LogRoot.Replace("'", "''")
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_text.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_vision.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_gaze.json") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ProjectRoot "event_fusion.json") -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $ProjectRoot -Filter "event_*.tmp" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

$terminals = @(
    @{
        Title = "ChefZeroWaste - Integrador"
        Log = "integrador.log"
        Command = "python IntegradorMultimodal.py --window-ms $WindowMs"
    },
    @{
        Title = "ChefZeroWaste - Head Tracking + Gestos"
        Log = "gaze_gestos.log"
        Command = "python vision\gaze_head_tracking.py"
    },
    @{
        Title = "ChefZeroWaste - Chat PLN"
        Log = "chat_pln.log"
        Command = "python main_chef_zero_waste.py"
    }
)

Write-Host "Proyecto: $ProjectRoot"
Write-Host "Ventana temporal del integrador: $WindowMs ms"
Write-Host "Ampliacion D: Head/Gaze Tracking + Gestos combinados"

Write-Host "Comprobando dependencias Python..."
python -c "import sklearn, cv2, mediapipe" 2>$null
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
    Write-Host "Orden de prueba:"
    Write-Host "1. Mira a la seccion de la pantalla que te interesa (ingredientes, pasos, receta)."
    Write-Host "2. Haz un gesto con la mano (pizca_sal, corte_cuchillo, sustituir)."
    Write-Host "3. Escribe la frase en Chat PLN y pulsa Enter."
    Write-Host "4. Mira la fusion en la ventana del Integrador: deberia incluir FOCO DE ATENCION."
}
