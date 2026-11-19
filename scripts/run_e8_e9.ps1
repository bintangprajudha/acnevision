# AcneVision - E8 (fine-tune tile 60 epoch dari E3) lalu E9 (training tile dari COCO, resep E3),
# masing-masing dievaluasi plain + SAHI 1728, lalu prediksi test dengan SAHI.
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_e8_e9.ps1
# Lanjut dari E9 (bila E8 sudah selesai):  ... -File scripts\run_e8_e9.ps1 -FromE9
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
# Estimasi (RTX 4060, ±390 dtk/epoch tile): E8 ±6,5 jam, E9 ±11 jam, eval + predict ±30 menit.
param([switch]$FromE9)
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

function Guard($w) {
    if (Test-Path $w) { Write-Host "Bobot sudah ada ($w); hapus/rename dulu bila ingin melatih ulang." -ForegroundColor Red; exit 1 }
}

function ValConf($run) {
    (Get-Content "..\runs\eval\$run\metrics.json" -Raw | ConvertFrom-Json).conf_selected_on_val
}

$cfg8 = "..\configs\yolov8l_960_tiles2_ft60.yaml"
$cfg9 = "..\configs\yolov8l_960_tiles2.yaml"
$e8 = "..\runs\detect\yolov8l_960_tiles2_ft60\weights\best.pt"
$e9 = "..\runs\detect\yolov8l_960_tiles2\weights\best.pt"

if (-not $FromE9) {
    Guard $e8
    Step "e8_smoke"     @("train.py", "--cfg", $cfg8, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "e8_smoke")
    Step "e8_train"     @("train.py", "--cfg", $cfg8)
    Step "eval_e8"      @("evaluate.py", "--weights", $e8)
    Step "eval_e8_sahi" @("evaluate.py", "--weights", $e8, "--mode", "sahi", "--slice", "1728")
}

Guard $e9
Step "e9_smoke"     @("train.py", "--cfg", $cfg9, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "e9_smoke")
Step "e9_train"     @("train.py", "--cfg", $cfg9)
Step "eval_e9"      @("evaluate.py", "--weights", $e9)
Step "eval_e9_sahi" @("evaluate.py", "--weights", $e9, "--mode", "sahi", "--slice", "1728")

# Prediksi test dengan SAHI pada conf yang dipilih di val
Step "predict_e8_sahi" @("predict.py", "--weights", $e8, "--source", "..\data\images\test", "--compare", "--sahi", "--slice", "1728",
                         "--conf", "$(ValConf 'yolov8l_960_tiles2_ft60_sahi1728')")
Step "predict_e9_sahi" @("predict.py", "--weights", $e9, "--source", "..\data\images\test", "--compare", "--sahi", "--slice", "1728",
                         "--conf", "$(ValConf 'yolov8l_960_tiles2_sahi1728')")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_e8_e9)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
