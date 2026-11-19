# AcneVision - B1: baseline YOLO11n @ 640 (jangkar pembanding SAT-AcneDet), smoke test -> training -> evaluasi plain.
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_baseline_b1.ps1
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
# Bobot COCO pretrained\yolo11n.pt (±5 MB) diunduh otomatis oleh Ultralytics saat smoke test.
# Estimasi (RTX 4060): training ±30-60 menit (model kecil, 640 px), eval ±3 menit.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"
$log = "..\runs\chain.log"
$smoke = Join-Path (Resolve-Path "..\runs").Path "smoke"

function Step($name, [string[]]$a) {
    $msg = "[$(Get-Date -Format HH:mm:ss)] START $name"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    & $py @a 2>&1 | Tee-Object -FilePath "..\runs\chain_$name.log"
    $code = $LASTEXITCODE
    $msg = "[$(Get-Date -Format HH:mm:ss)] END $name exit=$code"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    if ($code -ne 0) { Write-Host "GAGAL di langkah $name, berhenti." -ForegroundColor Red; exit $code }
}

$cfg = "..\configs\yolo11n_640.yaml"
$w = "..\runs\detect\yolo11n_640\weights\best.pt"
if (Test-Path $w) { Write-Host "Bobot sudah ada ($w); hapus/rename dulu bila ingin melatih ulang." -ForegroundColor Red; exit 1 }

Step "b1_smoke" @("train.py", "--cfg", $cfg, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "b1_smoke")
Step "b1_train" @("train.py", "--cfg", $cfg)
Step "eval_b1"  @("evaluate.py", "--weights", $w)
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_baseline_b1)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
