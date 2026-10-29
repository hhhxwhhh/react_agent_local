# Register / unregister the daily self-evolution task.
#
# Usage (PowerShell in the project root):
#   powershell -ExecutionPolicy Bypass -File .\register_scheduler.ps1
#   powershell -ExecutionPolicy Bypass -File .\register_scheduler.ps1 -Unregister
#
# The task runs only while the current user is logged on. If the PC is off or
# asleep at 12:00, the run is skipped instead of starting outside the cheap window.

param(
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$TaskName = 'ReActAgent-SelfEvolution-Noon'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Unregistered scheduled task: $TaskName"
    exit 0
}

$Python = (Get-Command python -ErrorAction Stop).Source
$Script = Join-Path $Root 'scheduler.py'
if (-not (Test-Path $Script)) {
    throw "Cannot find $Script"
}
$Argument = '-X utf8 "' + $Script + '" --start 12:00 --end 14:00 --max-generations 6'

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument $Argument `
    -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At '12:00'
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description 'Run the ReAct Agent self-evolution harness daily from 12:00 to 14:00.' `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Window: daily 12:00 - 14:00 (evolution.py commits or rolls back each generation)"
Write-Host "Logs: $Root\data\evolution\scheduler"
