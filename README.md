conda create -p myenv python==3.12
conda activate myenv/
pip install -r requirement.txt
streamlit run app.py
pip freeze > requirement1.txt
