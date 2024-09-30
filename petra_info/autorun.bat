@echo off
:: Activate the conda environment
call C:\Users\marin\anaconda3\condabin\activate.bat roboflow

cd "D:\Python\YOLOv8-and-GroundingDINO-for-Real-Time-License-Plate-Detection\petra_info"

:: Run your Python script
python D:\Python\YOLOv8-and-GroundingDINO-for-Real-Time-License-Plate-Detection\petra_info\async.py

pause