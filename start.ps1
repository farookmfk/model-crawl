# Windows launcher: creates the virtualenv on first run, then starts Model Crawl and opens the browser.
#   $env:MODEL_CRAWL_NO_BROWSER = "1"  to skip opening the browser
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv\.installed)) {
    python -m venv .venv
    .\.venv\Scripts\python -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    New-Item .venv\.installed -ItemType File | Out-Null
}
$port = if ($env:MODEL_CRAWL_PORT) { $env:MODEL_CRAWL_PORT } else { "8765" }
if (-not $env:MODEL_CRAWL_NO_BROWSER) { Start-Process "http://127.0.0.1:$port" }
.\.venv\Scripts\python server.py
