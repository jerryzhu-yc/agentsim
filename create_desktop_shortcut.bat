@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 在桌面创建 AgentSim 快捷方式（指向 start.bat，图标用仓库内 agentsim.ico）
rem 使用系统桌面真实路径（兼容 OneDrive 重定向桌面）

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\AgentSim.lnk'); $lnk.TargetPath = (Get-Location).Path + '\start.bat'; $lnk.WorkingDirectory = (Get-Location).Path; $lnk.IconLocation = (Get-Location).Path + '\agentsim.ico'; $lnk.Description = 'AgentSim 机器人仿真工作台'; $lnk.Save(); Write-Host ('已创建桌面快捷方式: ' + $lnk.FullName)"

if errorlevel 1 (
    echo [错误] 快捷方式创建失败（PowerShell 不可用或被策略拦截）。
) else (
    echo 双击桌面上的 AgentSim 图标即可启动 GUI。
)
pause
