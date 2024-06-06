# emea-swex-camp-langchain
EMEA SWEX camp AI workshop

## Create a .env file with following content

```
SERPER_API_KEY={serper_api_key}
OPENAI_API_KEY={open_api_key}
```

## Generate a python venv
First make sure you are using python 3.12
then run the following commands (on mac and linux)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app
To run the app run
```
streamlit run main.py
```
