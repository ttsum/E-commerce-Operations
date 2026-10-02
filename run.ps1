$ErrorActionPreference = 'Stop'
$env:PYTHONNOUSERSITE = '1'
& 'D:\ProgramData\Anaconda3\python.exe' (Join-Path $PSScriptRoot 'run_pipeline.py') @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
