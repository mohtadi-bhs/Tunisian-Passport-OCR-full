@echo off
title Label Studio Launcher

REM =====================================================
REM Activate virtual environment
REM =====================================================
call C:\Users\User\label_studio_env\Scripts\activate.bat

REM =====================================================
REM Enable local file serving
REM =====================================================
set LOCAL_FILES_SERVING_ENABLED=true


set LOCAL_FILES_DOCUMENT_ROOT=C:\Users\User\Desktop

REM =====================================================
REM Optional settings
REM =====================================================
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

REM =====================================================
REM Move to safe directory
REM =====================================================
cd /d C:\Users\User

REM =====================================================
REM Start Label Studio
REM =====================================================
label-studio start

REM =====================================================
REM Keep terminal open if error happens
REM =====================================================
pause