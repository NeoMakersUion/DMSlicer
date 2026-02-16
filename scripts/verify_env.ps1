Param(
    [string]$Root = ".",
    [switch]$RunTests = $true
)

$ErrorActionPreference = "Stop"

function Find-Projects {
    param([string]$Base)
    Get-ChildItem -Path $Base -Recurse -Filter "pyproject.toml" | ForEach-Object { $_.Directory.FullName } | Sort-Object -Unique
}

function Verify-Project {
    param([string]$Path)
    Push-Location $Path
    try {
        uv --version | Out-Null
        uv lock --check | Out-Null
        uv sync | Out-Null
        $py = uv run -- python -c "import sys; print(sys.version)" 2>&1
        $okPy = $py -match "^3\.12\."
        $tests = ""
        $testsOk = $false
        if ($RunTests) {
            try {
                $tests = uv run -- pytest -q
                $testsOk = $LASTEXITCODE -eq 0
            } catch {
                $tests = $_.Exception.Message
                $testsOk = $false
            }
        }
        return [PSCustomObject]@{
            Path = $Path
            Python = ($py -join "`n").Trim()
            PythonOk = $okPy
            TestsOk = $testsOk
            TestsOutput = ($tests -join "`n").Trim()
        }
    } finally {
        Pop-Location
    }
}

$projects = Find-Projects -Base $Root
if (-not $projects) { $projects = @((Resolve-Path $Root).Path) }

$results = @()
foreach ($p in $projects) {
    $results += Verify-Project -Path $p
}

$outDir = Join-Path (Resolve-Path $Root) "reports"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$outFile = Join-Path $outDir "verify_env_results.csv"
$results | Export-Csv -Path $outFile -NoTypeInformation -Encoding UTF8
Write-Host "Report written to: $outFile"

