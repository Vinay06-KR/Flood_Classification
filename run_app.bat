@echo off
REM Set PYTHONPATH to project root and run Streamlit
set PYTHONPATH=%CD%
streamlit run app/app.py
