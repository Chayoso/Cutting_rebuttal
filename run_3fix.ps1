$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
$fruits = @("pear", "tomato", "lemon")
$logDir = "sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$mainLog = Join-Path $logDir "fix3_run.log"
"3-fruit FIX run started at $(Get-Date)" | Out-File -FilePath $mainLog -Encoding utf8
$t0 = Get-Date
foreach ($f in $fruits) {
    $cfg = "configs/fruits/$f.yaml"
    $tFruit = Get-Date
    "[$(Get-Date -Format HH:mm:ss)] [START] $f" | Add-Content -Path $mainLog -Encoding utf8
    & $py -u run.py --config $cfg --headless --run-sim --show-board --rebuild-sdf --log-dir "logs/single_cut/$f" *> "$logDir/$f.fix3.log"
    $rc = $LASTEXITCODE
    $dur = [int]((Get-Date) - $tFruit).TotalSeconds
    "[$(Get-Date -Format HH:mm:ss)] [DONE]  $f rc=$rc ${dur}s" | Add-Content -Path $mainLog -Encoding utf8
}
$totalMin = [int](((Get-Date) - $t0).TotalMinutes)
"3-fruit FIX finished at $(Get-Date) -- total ${totalMin}min" | Add-Content -Path $mainLog -Encoding utf8
