# AcneVision - E6/E7: resep E3 dan E4 dengan arsitektur YOLO11l, lalu evaluasi plain + SAHI dan prediksi test.
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_yolo11.ps1
# Lanjut dari E7 (bila E6 sudah selesai):  ... -File scripts\run_yolo11.ps1 -FromE7
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
# Bobot COCO pretrained\yolo11l.pt (±51 MB) diunduh otomatis oleh Ultralytics saat smoke test pertama.
# Estimasi (RTX 4060): E6 training ±1,5-2,2 jam, E7 ±45-65 menit, eval ±20 menit, predict ±10 menit.
param([switch]$FromE7)
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

$cfg6 = "..\configs\yolo11l_960.yaml"
$cfg7 = "..\configs\yolo11l_960_tiles2_ft.yaml"
$e6 = "..\runs\detect\yolo11l_960\weights\best.pt"
$e7 = "..\runs\detect\yolo11l_960_tiles2_ft\weights\best.pt"

if (-not $FromE7) {
    Guard $e6
    Step "e6_smoke"     @("train.py", "--cfg", $cfg6, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "e6_smoke")
    Step "e6_train"     @("train.py", "--cfg", $cfg6)
    Step "eval_e6"      @("evaluate.py", "--weights", $e6)
    Step "eval_e6_sahi" @("evaluate.py", "--weights", $e6, "--mode", "sahi", "--slice", "1728")
}

Guard $e7
Step "e7_smoke"     @("train.py", "--cfg", $cfg7, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "e7_smoke")
Step "e7_train"     @("train.py", "--cfg", $cfg7)
Step "eval_e7"      @("evaluate.py", "--weights", $e7)
Step "eval_e7_sahi" @("evaluate.py", "--weights", $e7, "--mode", "sahi", "--slice", "1728")

# Prediksi test dengan conf yang dipilih di val (E6 plain, E7 SAHI)
Step "predict_e6"      @("predict.py", "--weights", $e6, "--source", "..\data\images\test", "--compare", "--conf", "$(ValConf 'yolo11l_960')")
Step "predict_e7_sahi" @("predict.py", "--weights", $e7, "--source", "..\data\images\test", "--compare", "--sahi", "--slice", "1728",
                         "--conf", "$(ValConf 'yolo11l_960_tiles2_ft_sahi1728')")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_yolo11)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
