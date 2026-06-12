# Spawn an executable and poll process-tree RSS every 100ms.
# Windows: Rscript may fork helpers; peak RSS must include the full tree
# (Rterm / rsession / python children), not just the launcher stub.
#
# Usage from R verify.R:
#   system2("powershell.exe", c("-NoProfile", "-ExecutionPolicy", "Bypass",
#     "-File", <this_script>, <stats_file>, <exe>, <exe_args>...))

if ($args.Count -lt 2) {
    [Console]::Error.WriteLine("win_peak_wrapper: expected <stats_file> <exe> [args...]; got $($args.Count) arg(s)")
    exit 2
}

$StatsFile = $args[0]
$Exe = $args[1]
$ExeArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }

if (-not $env:PROCESSOR_ARCHITECTURE) {
    $env:PROCESSOR_ARCHITECTURE = "AMD64"
}

function Get-DescendantPids {
    param(
        [int]$RootPid,
        [array]$AllProcs
    )
    $tree = New-Object System.Collections.Generic.HashSet[int]
    [void]$tree.Add($RootPid)
    $queue = New-Object System.Collections.Queue
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $procId = [int]$queue.Dequeue()
        foreach ($c in ($AllProcs | Where-Object { $_.ParentProcessId -eq $procId })) {
            $cpid = [int]$c.ProcessId
            if ($tree.Add($cpid)) { $queue.Enqueue($cpid) }
        }
    }
    return $tree
}

function Update-PeakFromTree {
    param(
        [System.Collections.Generic.HashSet[int]]$Pids
    )
    foreach ($procId in $Pids) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            if ($proc.WorkingSet64 -gt $script:peakWs) { $script:peakWs = $proc.WorkingSet64 }
            if ($proc.PeakWorkingSet64 -gt $script:peakWs) { $script:peakWs = $proc.PeakWorkingSet64 }
        } catch {
            # process exited between snapshot and Get-Process
        }
    }
}

$p = if ($ExeArgs.Count -gt 0) {
    Start-Process -FilePath $Exe -ArgumentList $ExeArgs -PassThru -NoNewWindow
} else {
    Start-Process -FilePath $Exe -PassThru -NoNewWindow
}

$script:peakWs = 0L
while (!$p.HasExited) {
    try {
        $p.Refresh()
        if ($p.WorkingSet64 -gt $script:peakWs) { $script:peakWs = $p.WorkingSet64 }
        if ($p.PeakWorkingSet64 -gt $script:peakWs) { $script:peakWs = $p.PeakWorkingSet64 }
    } catch {}
    $allProcs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    if ($allProcs.Count -gt 0) {
        $tree = Get-DescendantPids -RootPid $p.Id -AllProcs $allProcs
        Update-PeakFromTree -Pids $tree
    }
    Start-Sleep -Milliseconds 100
}

try {
    $p.Refresh()
    if ($p.WorkingSet64 -gt $script:peakWs) { $script:peakWs = $p.WorkingSet64 }
    if ($p.PeakWorkingSet64 -gt $script:peakWs) { $script:peakWs = $p.PeakWorkingSet64 }
    $allProcs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    if ($allProcs.Count -gt 0) {
        $tree = Get-DescendantPids -RootPid $p.Id -AllProcs $allProcs
        Update-PeakFromTree -Pids $tree
    }
} catch {}

$peak = $script:peakWs

"PEAK_WS_BYTES=$peak" | Out-File -FilePath $StatsFile -Encoding ascii
exit $p.ExitCode
