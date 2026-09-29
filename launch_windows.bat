@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab notebooks\Elder_Mist_RL_Lab.ipynb
