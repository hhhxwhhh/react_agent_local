# 注册 / 注销每天的自动自进化任务。
#
# 用法（在项目根目录打开 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File .\register_scheduler.ps1
#   powershell -ExecutionPolicy Bypass -File .\register_scheduler.ps1 -Unregister
#
# 任务只在当前用户登录时运行；如果 12:00 时电脑关机或休眠，当天不会补跑，
# 避免在便宜时段之外启动。

param(
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$TaskName = 'ReActAgent-SelfEvolution-Noon'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "已注销计划任务：$TaskName"
    exit 0
}

$Python = (Get-Command python -ErrorAction Stop).Source
$Script = Join-Path $Root 'scheduler.py'
if (-not (Test-Path $Script)) {
    throw "找不到 $Script"
}
$Argument = '-X utf8 "' + $Script + '" --start 12:00 --end 14:00 --max-generations 4'

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
    -Description '每天 12:00-14:00 逐代运行 ReAct Agent 自进化 Harness' `
    -Force | Out-Null

Write-Host "已注册计划任务：$TaskName"
Write-Host "运行时间：每天 12:00 - 14:00（每代由 evolution.py 自行提交/回滚）"
Write-Host "日志目录：$Root\data\evolution\scheduler"
