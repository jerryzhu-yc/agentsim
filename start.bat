@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem AgentSim 桌面 GUI 快速启动（Windows 双击入口）
rem Python 选择优先级：AGENTSIM_PYTHON 环境变量 > 仓库内 .venv > PATH 里的 python > py 启动器

set "PYEXE=%AGENTSIM_PYTHON%"
if not defined PYEXE if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE (
    where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    where py >nul 2>nul && set "PYEXE=py -3"
)
if not defined PYEXE (
    echo [错误] 未找到 Python。请安装 Python 3.10+ 并勾选 "Add python.exe to PATH"，
    echo        或设置环境变量 AGENTSIM_PYTHON 指向 python.exe 的完整路径。
    pause
    exit /b 1
)

echo 用 [%PYEXE%] 启动 AgentSim GUI ...
%PYEXE% -m agentsim gui
if errorlevel 1 (
    echo.
    echo [提示] GUI 异常退出。若上方报 ModuleNotFoundError，请先安装依赖：
    echo        pip install -r requirements.txt
    pause
)
