# Local release smoke for the public autozyme checkout.
#
# This is intentionally stricter than the normal framework unit tests in one
# way: Python packages are built into wheels and R is built into a source
# tarball before testing, so the exercised code path matches a user install.
# It is intentionally lighter than Tier B/C in another way: it does not install
# every heavy upstream package. Installed upstreams must activate cleanly;
# missing upstreams must be reported cleanly instead of crashing.

[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "python",
    [string]$R = "R.exe",
    [string]$Rscript = "Rscript.exe",
    [int]$RSmokeTimeoutSeconds = 300,
    [int]$RSmokePostReportGraceSeconds = 10,
    [switch]$SkipPython,
    [switch]$SkipR,
    [switch]$SkipCliE2E,
    [switch]$RunRTestthat,
    [int]$RTestthatTimeoutSeconds = 420,
    [int]$RTestthatBatchSize = 1,
    [switch]$KeepTemp
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if (-not $env:PROCESSOR_ARCHITECTURE) {
    $env:PROCESSOR_ARCHITECTURE = "AMD64"
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Get-TextFileTail {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Tail = 80
    )
    if (!(Test-Path -LiteralPath $Path)) {
        return ""
    }
    return (Get-Content -LiteralPath $Path -Tail $Tail -ErrorAction SilentlyContinue | Out-String)
}

function Stop-ProcessesContainingCommandLine {
    param([Parameter(Mandatory = $true)][string]$Needle)
    $matches = @(Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($Needle) })
    foreach ($match in $matches) {
        try {
            Stop-Process -Id $match.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Failed to stop lingering process $($match.ProcessId): $($_.Exception.Message)"
        }
    }
}

function Test-RSmokeReportPassed {
    param([Parameter(Mandatory = $true)][string]$ReportPath)
    if (!(Test-Path -LiteralPath $ReportPath)) {
        return $false
    }
    try {
        $rows = @(Import-Csv -LiteralPath $ReportPath -Delimiter "`t")
    } catch {
        return $false
    }
    if ($rows.Count -eq 0) {
        return $false
    }
    foreach ($row in $rows) {
        if ($row.error -and $row.error.Trim().Length -gt 0) {
            return $false
        }
        $installed = ($row.upstream_installed -eq "TRUE")
        if ($installed) {
            if ($row.activate_result -ne "TRUE") {
                return $false
            }
            if ($row.status_after_activate -ne "active") {
                return $false
            }
            if ($row.status_after_restore -ne "inactive") {
                return $false
            }
        } elseif ($row.activate_result -eq "TRUE") {
            return $false
        }
    }
    return $true
}

function Invoke-RSmokeScript {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$ReportPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][int]$PostReportGraceSeconds
    )

    $stdout = Join-Path $WorkingDirectory "r_patch_smoke.stdout.log"
    $stderr = Join-Path $WorkingDirectory "r_patch_smoke.stderr.log"
    $argText = '"' + $ScriptPath + '"'
    $proc = Start-Process -FilePath $Exe -ArgumentList $argText -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $started = Get-Date
    $passedReportAt = $null

    while (!$proc.HasExited) {
        if (Test-RSmokeReportPassed $ReportPath) {
            if ($null -eq $passedReportAt) {
                $passedReportAt = Get-Date
            } elseif (((Get-Date) - $passedReportAt).TotalSeconds -ge $PostReportGraceSeconds) {
                Stop-ProcessesContainingCommandLine $ScriptPath
                Write-Warning "R smoke report passed, but Rscript did not exit within $PostReportGraceSeconds seconds after writing the report; stopped lingering process."
                return
            }
        }

        if (((Get-Date) - $started).TotalSeconds -ge $TimeoutSeconds) {
            Stop-ProcessesContainingCommandLine $ScriptPath
            if (Test-RSmokeReportPassed $ReportPath) {
                Write-Warning "R smoke report passed, but Rscript exceeded $TimeoutSeconds seconds; stopped lingering process."
                return
            }
            $outTail = Get-TextFileTail $stdout
            $errTail = Get-TextFileTail $stderr
            throw "R patch smoke timed out after $TimeoutSeconds seconds.`nSTDOUT tail:`n$outTail`nSTDERR tail:`n$errTail"
        }

        Start-Sleep -Seconds 2
        $proc.Refresh()
    }

    try {
        $proc.WaitForExit()
    } catch {
        # Some Windows Rscript runs can finish while leaving an incomplete
        # Process object. The structured report below is the source of truth.
    }

    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) {
        if (Test-RSmokeReportPassed $ReportPath) {
            Write-Warning "R patch smoke wrote a passing report but Rscript did not expose an exit code; treating the structured report as passing."
            return
        }
        $outTail = Get-TextFileTail $stdout
        $errTail = Get-TextFileTail $stderr
        throw "R patch smoke exited without an exit code and did not write a passing report.`nSTDOUT tail:`n$outTail`nSTDERR tail:`n$errTail"
    }

    if ($exitCode -ne 0) {
        $outTail = Get-TextFileTail $stdout
        $errTail = Get-TextFileTail $stderr
        throw "R patch smoke failed with exit code $exitCode.`nSTDOUT tail:`n$outTail`nSTDERR tail:`n$errTail"
    }
    if (!(Test-Path -LiteralPath $ReportPath)) {
        $outTail = Get-TextFileTail $stdout
        $errTail = Get-TextFileTail $stderr
        throw "R patch smoke did not write report: $ReportPath`nSTDOUT tail:`n$outTail`nSTDERR tail:`n$errTail"
    }
    if (!(Test-RSmokeReportPassed $ReportPath)) {
        $outTail = Get-TextFileTail $stdout
        $errTail = Get-TextFileTail $stderr
        throw "R patch smoke report contains failures: $ReportPath`nSTDOUT tail:`n$outTail`nSTDERR tail:`n$errTail"
    }
}

function Invoke-RTestthatBatch {
    param(
        [Parameter(Mandatory = $true)][string]$Runner,
        [Parameter(Mandatory = $true)][string]$PackageDir,
        [Parameter(Mandatory = $true)][string]$RscriptPath,
        [Parameter(Mandatory = $true)][string]$RLibPath,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][string[]]$TestFiles,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Runner,
        "-PackageDir", $PackageDir,
        "-Rscript", $RscriptPath,
        "-RLib", $RLibPath,
        "-TimeoutSeconds", [string]$TimeoutSeconds,
        "-TestFiles"
    ) + $TestFiles
    Invoke-Native -Exe $powershellExe -Arguments $args -WorkingDirectory $WorkingDirectory
}

function Get-VenvPython {
    param([Parameter(Mandatory = $true)][string]$Venv)
    $candidates = @(
        (Join-Path $Venv "Scripts\python.exe"),
        (Join-Path $Venv "bin\python")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "Could not find python executable under $Venv"
}

function Get-SingleWheel {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$PackagePrefix
    )
    $wheels = @(Get-ChildItem -LiteralPath $Directory -Filter "$PackagePrefix*.whl" -File |
        Sort-Object LastWriteTime -Descending)
    if ($wheels.Count -ne 1) {
        throw "Expected exactly one wheel matching $PackagePrefix*.whl in $Directory; found $($wheels.Count)"
    }
    return $wheels[0].FullName
}

function Remove-GeneratedPythonBuildState {
    $paths = @(
        (Join-Path $RepoRoot "autozyme_py\src\autozyme.egg-info"),
        (Join-Path $RepoRoot "autozyme_py\build"),
        (Join-Path $RepoRoot "autozyme_cli\autozyme_cli.egg-info"),
        (Join-Path $RepoRoot "autozyme_cli\build")
    )
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

function Count-Files {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Filter
    )
    if (!(Test-Path -LiteralPath $Path)) {
        return 0
    }
    return @(Get-ChildItem -LiteralPath $Path -Recurse -File -Filter $Filter).Count
}

function Assert-ReleaseSourceSanity {
    Write-Step "Release source sanity"
    $pyRoot = Join-Path $RepoRoot "autozyme_py\src\autozyme"
    $rRoot = Join-Path $RepoRoot "autozyme_r\inst\patches"

    $pyFinalized = Count-Files $pyRoot "speedups_finalized.tsv"
    $rFinalized = Count-Files $rRoot "speedups_finalized.tsv"
    $pyRaw = Count-Files $pyRoot "speedups.tsv"
    $rRaw = Count-Files $rRoot "speedups.tsv"
    $rLegacyPatchFiles = 0
    if (Test-Path -LiteralPath $rRoot) {
        $rLegacyPatchFiles = @(Get-ChildItem -LiteralPath $rRoot -File -Filter "*.R").Count
    }
    $rLegacySpeedupsDir = Test-Path -LiteralPath (Join-Path $RepoRoot "autozyme_r\inst\speedups")
    $datasetsDir = Test-Path -LiteralPath (Join-Path $RepoRoot "datasets")
    $scblasDirs = @(Get-ChildItem -LiteralPath $RepoRoot -Recurse -Directory -Filter "scblas" -ErrorAction SilentlyContinue)

    $templates = Get-Item -LiteralPath (Join-Path $RepoRoot "autozyme_cli\templates") -ErrorAction Stop
    $templatesIsLink = $false
    if ($null -ne $templates.LinkType -and $templates.LinkType.Length -gt 0) {
        $templatesIsLink = $true
    }

    $reflectionRoot = Join-Path $RepoRoot "reflections"
    $reflectionFiles = 0
    if (Test-Path -LiteralPath $reflectionRoot) {
        $reflectionFiles = @(Get-ChildItem -LiteralPath $reflectionRoot -Recurse -File -ErrorAction SilentlyContinue).Count
    }

    Write-Host "Python finalized TSVs: $pyFinalized"
    Write-Host "R finalized TSVs:      $rFinalized"
    Write-Host "Python raw speedups:   $pyRaw"
    Write-Host "R raw speedups:        $rRaw"
    Write-Host "R legacy patch files:  $rLegacyPatchFiles"
    Write-Host "R legacy speedups dir: $rLegacySpeedupsDir"
    Write-Host "datasets dir exists:   $datasetsDir"
    Write-Host "scblas dir count:      $($scblasDirs.Count)"
    Write-Host "CLI templates symlink: $templatesIsLink"
    Write-Host "reflection file count: $reflectionFiles"

    if ($pyFinalized -ne 23) { throw "Expected 23 Python speedups_finalized.tsv files; found $pyFinalized" }
    if ($rFinalized -ne 23) { throw "Expected 23 R speedups_finalized.tsv files; found $rFinalized" }
    if ($pyRaw -ne 0) { throw "Raw Python speedups.tsv files must not ship in release" }
    if ($rRaw -ne 0) { throw "Raw R speedups.tsv files must not ship in release" }
    if ($rLegacyPatchFiles -ne 0) { throw "Legacy autozyme_r/inst/patches/*.R files must not ship in release" }
    if ($rLegacySpeedupsDir) { throw "Legacy autozyme_r/inst/speedups directory must not ship in release" }
    if ($datasetsDir) { throw "datasets/ must not ship in release checkout" }
    if ($scblasDirs.Count -ne 0) { throw "scblas directories must not ship in release checkout" }
    if ($templatesIsLink) { throw "autozyme_cli/templates must be a real directory, not a symlink" }
}

function Run-PythonSmoke {
    param([Parameter(Mandatory = $true)][string]$TmpRoot)

    Write-Step "Build and install Python wheels"
    $venv = Join-Path $TmpRoot "py-smoke"
    Invoke-Native $Python @("-m", "venv", $venv)
    $py = Get-VenvPython $venv

    Invoke-Native $py @("-m", "pip", "install", "--upgrade", "pip", "build", "pytest", "twine")

    $pyDist = Join-Path $TmpRoot "dist\autozyme_py"
    $cliDist = Join-Path $TmpRoot "dist\autozyme_cli"
    New-Item -ItemType Directory -Force -Path $pyDist, $cliDist | Out-Null

    try {
        $pyBuildArgs = @("-m", "build", "--wheel", "--sdist", "--outdir", $pyDist)
        $cliBuildArgs = @("-m", "build", "--wheel", "--sdist", "--outdir", $cliDist)
        Invoke-Native -Exe $py -Arguments $pyBuildArgs -WorkingDirectory (Join-Path $RepoRoot "autozyme_py")
        Invoke-Native -Exe $py -Arguments $cliBuildArgs -WorkingDirectory (Join-Path $RepoRoot "autozyme_cli")
    } finally {
        Remove-GeneratedPythonBuildState
    }

    Write-Step "Check Python distribution metadata"
    $distFiles = @(
        @(Get-ChildItem -LiteralPath $pyDist -File) +
        @(Get-ChildItem -LiteralPath $cliDist -File)
    ) | ForEach-Object { $_.FullName }
    Invoke-Native -Exe $py -Arguments (@("-m", "twine", "check") + $distFiles)

    $autozymeWheel = Get-SingleWheel $pyDist "autozyme-"
    $cliWheel = Get-SingleWheel $cliDist "autozyme_cli-"
    Invoke-Native $py @("-m", "pip", "install", $autozymeWheel, $cliWheel)

    Write-Step "Python patch install smoke"
    $pyReport = Join-Path $TmpRoot "python_patch_smoke.json"
    $pyScript = Join-Path $TmpRoot "python_patch_smoke.py"
    @'
import json
import os
import sys

import autozyme
from autozyme._core import _probe_patch_installed

patches = list(autozyme.list_patches())
if not patches:
    raise SystemExit("no Python patches discovered after wheel install")

rows = []
failures = []

for name in patches:
    entry = {
        "patch": name,
        "speedup_rows": len(autozyme.speedups(name)),
    }
    installed, reason = _probe_patch_installed(name)
    entry["upstream_installed"] = bool(installed)
    entry["probe_reason"] = reason or ""
    try:
        result = autozyme.activate(name)
        entry["activate_result"] = bool(result)
        if installed:
            if result is not True:
                failures.append(f"{name}: installed upstream but activate returned {result!r}")
            if autozyme.status().get(name) != "active":
                failures.append(f"{name}: status is not active after activate")
            autozyme.restore(name)
            if autozyme.status().get(name) != "inactive":
                failures.append(f"{name}: status is not inactive after restore")
        else:
            if result is not False:
                failures.append(f"{name}: missing upstream should return False, got {result!r}")
    except Exception as exc:
        entry["error"] = repr(exc)
        failures.append(f"{name}: activate/restore raised {exc!r}")
    finally:
        try:
            autozyme.restore_all()
        except Exception as exc:
            failures.append(f"{name}: restore_all raised {exc!r}")
    rows.append(entry)

report = {
    "patch_count": len(patches),
    "installed_upstream_count": sum(1 for row in rows if row["upstream_installed"]),
    "missing_upstream_count": sum(1 for row in rows if not row["upstream_installed"]),
    "speedup_file_backed_count": sum(1 for row in rows if row["speedup_rows"] > 0),
    "rows": rows,
    "failures": failures,
}

path = os.environ.get("AUTOZYME_PY_SMOKE_REPORT")
if path:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

print(json.dumps(report, indent=2, sort_keys=True))
if failures:
    raise SystemExit("Python patch smoke failed")
'@ | Set-Content -LiteralPath $pyScript -Encoding ASCII

    $oldQuiet = $env:AUTOZYME_QUIET
    $oldReport = $env:AUTOZYME_PY_SMOKE_REPORT
    try {
        $env:AUTOZYME_QUIET = "1"
        $env:AUTOZYME_PY_SMOKE_REPORT = $pyReport
        Invoke-Native $py @($pyScript)
    } finally {
        $env:AUTOZYME_QUIET = $oldQuiet
        $env:AUTOZYME_PY_SMOKE_REPORT = $oldReport
    }
    if (!(Test-Path -LiteralPath $pyReport)) {
        throw "Python patch smoke did not write report: $pyReport"
    }

    Write-Step "CLI installed command smoke"
    Invoke-Native $py @("-m", "zyme", "--help")
    Invoke-Native (Join-Path (Split-Path $py -Parent) "zyme.exe") @("--help")

    if (!$SkipCliE2E) {
        Write-Step "CLI end-to-end pytest smoke"
        $cliTestArgs = @("-m", "pytest", "-q", (Join-Path $RepoRoot "autozyme_cli\tests\test_cli_e2e.py"))
        Invoke-Native -Exe $py -Arguments $cliTestArgs -WorkingDirectory $TmpRoot
    }

    Write-Host "Python patch report: $pyReport"
}

function Run-RSmoke {
    param([Parameter(Mandatory = $true)][string]$TmpRoot)

    Write-Step "Build and install R source tarball"
    $rBuildDir = Join-Path $TmpRoot "r-build"
    $rLib = Join-Path $TmpRoot "r-lib"
    New-Item -ItemType Directory -Force -Path $rBuildDir, $rLib | Out-Null

    $rBuildArgs = @("CMD", "build", "--no-manual", (Join-Path $RepoRoot "autozyme_r"))
    Invoke-Native -Exe $R -Arguments $rBuildArgs -WorkingDirectory $rBuildDir
    $tarballs = @(Get-ChildItem -LiteralPath $rBuildDir -File -Filter "autozyme_*.tar.gz" | Sort-Object LastWriteTime -Descending)
    if ($tarballs.Count -ne 1) {
        throw "Expected one R source tarball in $rBuildDir; found $($tarballs.Count)"
    }
    $rInstallArgs = @("CMD", "INSTALL", "-l", $rLib, $tarballs[0].FullName)
    Invoke-Native -Exe $R -Arguments $rInstallArgs

    Write-Step "R patch install smoke"
    $rReport = Join-Path $TmpRoot "r_patch_smoke.tsv"
    $rSmokeScript = Join-Path $TmpRoot "r_patch_smoke.R"
    @'
if (.Platform$OS.type == "windows" &&
    !nzchar(Sys.getenv("PROCESSOR_ARCHITECTURE", unset = ""))) {
  Sys.setenv(PROCESSOR_ARCHITECTURE = "AMD64")
}

lib <- Sys.getenv("AUTOZYME_SMOKE_R_LIB", unset = "")
if (nzchar(lib)) {
  .libPaths(c(lib, .libPaths()))
}

suppressPackageStartupMessages(library(autozyme))

patches <- autozyme::list_patches()
if (!length(patches)) {
  stop("no R patches discovered after package install", call. = FALSE)
}

rows <- list()
failures <- character(0)

for (name in patches) {
  probe <- autozyme:::.probe_patch_installed(name)
  installed <- isTRUE(probe$installed)
  reason <- if (!is.null(probe$error) && !is.na(probe$error)) probe$error else ""
  speedup_rows <- tryCatch(nrow(autozyme::speedups(name)), error = function(e) NA_integer_)
  error <- ""
  activate_result <- NA
  status_after_activate <- ""
  status_after_restore <- ""

  tryCatch({
    result <- autozyme::activate(name)
    activate_result <- isTRUE(result)
    if (installed) {
      if (!isTRUE(result)) {
        failures <- c(failures, sprintf("%s: installed upstream but activate returned FALSE", name))
      }
      st <- autozyme::status()
      status_after_activate <- unname(st[name])
      if (!identical(status_after_activate, "active")) {
        failures <- c(failures, sprintf("%s: status is not active after activate", name))
      }
      autozyme::restore(name)
      st <- autozyme::status()
      status_after_restore <- unname(st[name])
      if (!identical(status_after_restore, "inactive")) {
        failures <- c(failures, sprintf("%s: status is not inactive after restore", name))
      }
    } else {
      if (isTRUE(result)) {
        failures <- c(failures, sprintf("%s: missing upstream should return FALSE", name))
      }
    }
  }, error = function(e) {
    error <<- conditionMessage(e)
    failures <<- c(failures, sprintf("%s: activate/restore raised: %s", name, error))
  }, finally = {
    try(autozyme::restore_all(), silent = TRUE)
  })

  rows[[length(rows) + 1L]] <- data.frame(
    patch = name,
    upstream_installed = installed,
    probe_reason = reason,
    activate_result = activate_result,
    status_after_activate = status_after_activate,
    status_after_restore = status_after_restore,
    speedup_rows = speedup_rows,
    error = error,
    stringsAsFactors = FALSE
  )
}

report <- do.call(rbind, rows)
utils::write.table(report, file = Sys.getenv("AUTOZYME_R_SMOKE_REPORT"),
                   sep = "\t", quote = FALSE, row.names = FALSE)
print(report)
cat(sprintf(
  "R patch smoke: %d patches, %d installed upstreams, %d missing upstreams\n",
  length(patches), sum(report$upstream_installed), sum(!report$upstream_installed)
))

if (length(failures)) {
  cat(paste(failures, collapse = "\n"), "\n", file = stderr())
  quit(save = "no", status = 1L, runLast = FALSE)
}
quit(save = "no", status = 0L, runLast = FALSE)
'@ | Set-Content -LiteralPath $rSmokeScript -Encoding ASCII

    $oldLib = $env:AUTOZYME_SMOKE_R_LIB
    $oldReport = $env:AUTOZYME_R_SMOKE_REPORT
    $oldQuiet = $env:AUTOZYME_QUIET
    try {
        $env:AUTOZYME_SMOKE_R_LIB = $rLib
        $env:AUTOZYME_R_SMOKE_REPORT = $rReport
        $env:AUTOZYME_QUIET = "1"
        Invoke-RSmokeScript -Exe $Rscript -ScriptPath $rSmokeScript -ReportPath $rReport `
            -WorkingDirectory $TmpRoot -TimeoutSeconds $RSmokeTimeoutSeconds `
            -PostReportGraceSeconds $RSmokePostReportGraceSeconds
    } finally {
        $env:AUTOZYME_SMOKE_R_LIB = $oldLib
        $env:AUTOZYME_R_SMOKE_REPORT = $oldReport
        $env:AUTOZYME_QUIET = $oldQuiet
    }
    if (!(Test-Path -LiteralPath $rReport)) {
        throw "R patch smoke did not write report: $rReport"
    }

    Write-Host "R patch report: $rReport"

    if ($RunRTestthat) {
        Write-Step "R testthat contracts"
        $runner = Join-Path $RepoRoot "scripts\local_r_testthat_files.ps1"
        if (!(Test-Path -LiteralPath $runner)) {
            throw "R testthat runner not found: $runner"
        }
        $testDir = Join-Path $RepoRoot "autozyme_r\tests\testthat"
        $testFiles = @(Get-ChildItem -LiteralPath $testDir -Filter "test-*.R" -File | Sort-Object Name | ForEach-Object { $_.Name })
        if ($testFiles.Count -eq 0) {
            throw "No R testthat files found under $testDir"
        }

        if ($RTestthatBatchSize -le 0 -or $RTestthatBatchSize -ge $testFiles.Count) {
            Write-Host "R testthat batch 1/1: $($testFiles.Count) files"
            Invoke-RTestthatBatch -Runner $runner -PackageDir (Join-Path $RepoRoot "autozyme_r") `
                -RscriptPath $Rscript -RLibPath $rLib -TimeoutSeconds $RTestthatTimeoutSeconds `
                -TestFiles $testFiles -WorkingDirectory $RepoRoot
        } else {
            $batchCount = [int][Math]::Ceiling($testFiles.Count / [double]$RTestthatBatchSize)
            for ($start = 0; $start -lt $testFiles.Count; $start += $RTestthatBatchSize) {
                $end = [Math]::Min($start + $RTestthatBatchSize - 1, $testFiles.Count - 1)
                $batch = @($testFiles[$start..$end])
                $batchIndex = [int][Math]::Floor($start / [double]$RTestthatBatchSize) + 1
                Write-Host "R testthat batch ${batchIndex}/${batchCount}: $($batch.Count) files"
                try {
                    Invoke-RTestthatBatch -Runner $runner -PackageDir (Join-Path $RepoRoot "autozyme_r") `
                        -RscriptPath $Rscript -RLibPath $rLib -TimeoutSeconds $RTestthatTimeoutSeconds `
                        -TestFiles $batch -WorkingDirectory $RepoRoot
                } catch {
                    throw "R testthat contracts failed in batch $batchIndex/$batchCount`: $($_.Exception.Message)"
                }
            }
        }
    }
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (!(Test-Path -LiteralPath (Join-Path $RepoRoot "autozyme_py"))) {
    throw "RepoRoot does not look like autozyme release checkout: $RepoRoot"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) "autozyme_release_smoke_$stamp"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

Write-Host "RepoRoot: $RepoRoot"
Write-Host "TempRoot: $tmpRoot"

try {
    Assert-ReleaseSourceSanity
    if (!$SkipPython) {
        Run-PythonSmoke $tmpRoot
    }
    if (!$SkipR) {
        Run-RSmoke $tmpRoot
    }
    Write-Step "Release smoke passed"
    Write-Host "Reports kept at: $tmpRoot"
} finally {
    if (!$KeepTemp) {
        Write-Host "Temp files are in $tmpRoot (use -KeepTemp to preserve on a future run)."
    }
}
