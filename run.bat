@echo off
chcp 65001 >nul
call conda activate kriging_gauss
python kriging_gauss.py
pause
