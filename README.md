# DeepSeek-R1
## Install Ollama according to the environment
- https://ollama.com/download

## Create the venv
- python -m venv venv
- souce venv/bin/activate
- pip install --upgrade pip

## Install Library
- pip install -r requirements.txt

## Run Ollama server and Download the DeepSeek Model
- ollama serve
- ollama run deepseek-r1:1.5b (Model Details : https://ollama.com/library/deepseek-r1)

## Pull the model and Run (Using Ollama Model)
- python run_deepseek_ollama.py --model 1.5b

## Pull the model and Run (Using Hugging Face Model)
- python run_deepseek_1.5b.py

### You can find model list in hugging face server
- python search_deepseek_model.py
