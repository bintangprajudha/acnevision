# AcneVision - E5: tile fine-tune dari B0, lalu evaluasi plain + SAHI dan prediksi test.
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_e5.ps1
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
# Estimasi (RTX 4060): smoke ±3 menit, training ±66 menit, eval plain ±10, eval SAHI ±15, predict ±15.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"
$log = "..\runs\chain.log"

function Step($name, [string[]]$a) {
    $msg = "[$(Get-Date -Format HH:mm:ss)] START $name"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    & $py @a 2>&1 | Tee-Object -FilePath "..\runs\chain_$name.log"
    $code = $LASTEXITCODE
    $msg = "[$(Get-Date -Format HH:mm:ss)] END $name exit=$code"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    if ($code -ne 0) { Write-Host "GAGAL di langkah $name, berhenti." -ForegroundColor Red; exit $code }
}

$cfg = "..\configs\yolov8l_960_default_tiles2_ft.yaml"
$e5 = "..\runs\detect\yolov8l_960_default_tiles2_ft\weights\best.pt"
if (Test-Path $e5) { Write-Host "Bobot E5 sudah ada ($e5); hapus/rename dulu bila ingin melatih ulang." -ForegroundColor Red; exit 1 }

Step "e5_smoke"     @("train.py", "--cfg", $cfg, "--epochs", "1", "--fraction", "0.02", "--project", (Join-Path (Resolve-Path "..\runs").Path "smoke"), "--name", "e5_smoke")
Step "e5_train"     @("train.py", "--cfg", $cfg)
Step "eval_e5"      @("evaluate.py", "--weights", $e5)
Step "eval_e5_sahi" @("evaluate.py", "--weights", $e5, "--mode", "sahi", "--slice", "1728")

# Prediksi test dengan conf yang dipilih di val oleh eval SAHI
$conf = (Get-Content "..\runs\eval\yolov8l_960_default_tiles2_ft_sahi1728\metrics.json" -Raw | ConvertFrom-Json).conf_selected_on_val
Step "predict_e5_sahi" @("predict.py", "--weights", $e5, "--source", "..\data\images\test", "--compare", "--sahi", "--slice", "1728", "--conf", "$conf")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_e5)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
