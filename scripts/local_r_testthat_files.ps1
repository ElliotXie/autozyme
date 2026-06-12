# Run autozyme_r testthat files one-by-one with a marker-based timeout.
#
# On this Windows host, some R/testthat/rlang combinations finish tests but
# keep the R process alive. Each generated R runner writes PASS/FAIL before
# sleeping; PowerShell then kills that process tree and records the marker.

[CmdletBinding()]
param(
    [string]$PackageDir = (Resolve-Path (Join-Path $PSScriptRoot "..\autozyme_r")).Path,
    [string]$Rscript = "Rscript.exe",
    [string]$RLib = "",
    [int]$TimeoutSeconds = 420,
    [string]$Pattern = "test-*.R",
    [string[]]$TestFiles = @()
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if (-not $env:PROCESSOR_ARCHITECTURE) {
    $env:PROCESSOR_ARCHITECTURE = "AMD64"
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$testDir = Join-Path $PackageDir "tests\testthat"
if (!(Test-Path -LiteralPath $testDir)) {
    throw "testthat directory not found: $testDir"
}

$logRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("autozyme_r_testthat_" + [System.DateTime]::Now.ToString("yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $logRoot | Out-Null
$summary = Join-Path $logRoot "summary.tsv"
Write-Host "summary=$summary"

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    if ($ProcessId -eq $PID) {
        return
    }
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Write-CurrentSummary {
    param([Parameter(Mandatory = $true)][object[]]$Rows)
    if ($Rows.Count -eq 0) {
        return
    }
    $Rows | Export-Csv -Path $summary -NoTypeInformation -Delimiter "`t"
}

function Get-PatchForTestFile {
    param([Parameter(Mandatory = $true)][string]$Name)
    switch -Regex ($Name) {
        '^test-contract-(bayesspace|cellchat|clusterprofiler|decontx|fgsea|infercnv|maftools|mast|nichenetr|rctd|scriabin|slingshot|tradeseq|vegan|wgcna)\.R$' {
            return $Matches[1]
        }
        '^test-contract-(FindAllMarkers|FindIntegrationAnchors|FindNeighbors|FindVariableFeatures|NormalizeData|RunCCA|RunPCA|ScaleData|SCTransform|seurat-s3-stragglers)\.R$' {
            return "seurat"
        }
        '^test-seurat-parameter-policy\.R$' {
            return "seurat"
        }
        default {
            return ""
        }
    }
}

if ($TestFiles.Count -gt 0) {
    $files = @()
    foreach ($item in $TestFiles) {
        $candidate = $item
        if (!(Test-Path -LiteralPath $candidate)) {
            $candidate = Join-Path $testDir $item
        }
        if (!(Test-Path -LiteralPath $candidate)) {
            throw "test file not found: $item"
        }
        $files += Get-Item -LiteralPath $candidate
    }
    $files = @($files | Sort-Object Name)
} else {
    $files = @(Get-ChildItem -Path $testDir -Filter $Pattern | Sort-Object Name)
}
if ($files.Count -eq 0) {
    throw "no test files matched $Pattern under $testDir"
}

$rows = @()
foreach ($file in $files) {
    $name = $file.Name
    $safe = $name -replace '[^A-Za-z0-9_.-]', '_'
    $out = Join-Path $logRoot ($safe + ".out.txt")
    $err = Join-Path $logRoot ($safe + ".err.txt")
    $runner = Join-Path $logRoot ($safe + ".runner.R")
    $marker = Join-Path $logRoot ($safe + ".status.txt")

    $libLine = ""
    if ($RLib -and $RLib.Trim().Length -gt 0) {
        $libLine = ".libPaths(c('$($RLib.Replace('\','/'))', .libPaths()))"
    }
    $patchName = Get-PatchForTestFile $name
    $patchLine = ""
    if ($patchName) {
        $patchLine = @"
.patch_name <- '$patchName'
.activated <- tryCatch(autozyme::activate(.patch_name), error = function(e) e)
if (inherits(.activated, 'error')) {
  stop(.activated)
}
rm(.patch_name, .activated)
"@
    }
    $expr = @"
if (.Platform`$OS.type == 'windows' &&
    !nzchar(Sys.getenv('PROCESSOR_ARCHITECTURE', unset = ''))) {
  Sys.setenv(PROCESSOR_ARCHITECTURE = 'AMD64')
}
Sys.setenv(AUTOZYME_TEST_SKIP_AUTO_ACTIVATE = '1')
$libLine
setwd('$($PackageDir.Replace('\','/'))')
library(testthat)
library(autozyme)
$patchLine
status <- 'PASS'
err <- NULL
tryCatch(
  testthat::test_file('$($file.FullName.Replace('\','/'))', reporter='summary'),
  error = function(e) {
    status <<- 'FAIL'
    err <<- conditionMessage(e)
  }
)
writeLines(c(status, if (is.null(err)) '' else err),
           con = '$($marker.Replace('\','/'))')
flush.console()
Sys.sleep(3600)
"@
    Set-Content -LiteralPath $runner -Value $expr -Encoding ASCII

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Rscript -ArgumentList @("--vanilla", $runner) `
        -RedirectStandardOutput $out -RedirectStandardError $err `
        -PassThru -WindowStyle Hidden
    $deadline = [System.DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([System.DateTime]::UtcNow -lt $deadline -and
           -not (Test-Path -LiteralPath $marker) -and
           -not $p.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    $sw.Stop()

    if (Test-Path -LiteralPath $marker) {
        $markerLines = Get-Content -LiteralPath $marker -ErrorAction SilentlyContinue
        $status = if ($markerLines.Count -gt 0) { $markerLines[0] } else { "FAIL" }
        $code = if ($status -eq "PASS") { 0 } else { 1 }
        $message = if ($markerLines.Count -gt 1) { ($markerLines[1..($markerLines.Count - 1)] -join "`n").Trim() } else { "" }
        Stop-ProcessTree -ProcessId $p.Id
    } elseif (-not $p.HasExited) {
        Stop-ProcessTree -ProcessId $p.Id
        $status = "TIMEOUT"
        $code = $null
        $message = "Timed out after $TimeoutSeconds seconds"
    } else {
        $code = $p.ExitCode
        $status = if ($code -eq 0) { "PASS" } else { "FAIL" }
        $message = ""
    }

    $row = [pscustomobject]@{
        file = $name
        status = $status
        exit_code = $code
        seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        message = $message
        stdout = $out
        stderr = $err
    }
    $rows += $row
    Write-CurrentSummary -Rows $rows
    Write-Host ("{0}`t{1}`t{2}s" -f $status, $name, $row.seconds)
}

Write-CurrentSummary -Rows $rows
Write-Host "summary=$summary"
if ($rows.status -contains "FAIL" -or $rows.status -contains "TIMEOUT") {
    exit 1
}
exit 0
