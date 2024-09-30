@echo off
:: Activate the conda environment
call C:\Users\marin\anaconda3\condabin\activate.bat roboflow

cd "D:\Python\Vehicle-Detection-and-Tracking-System\petra_info"

:: Run your Python script
python D:\Python\Vehicle-Detection-and-Tracking-System\petra_info\async.py

pause