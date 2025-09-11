# An AI Driven Harmonized Data Ingestion: Vision For Seamless Cross-Domain Data Integration

## ⚙️ Setup
Install OLLAMA on your laptop from their official [website](https://ollama.com/download), more infromations can also be found at their GitHub [repo](https://github.com/ollama/ollama). Choose your OS, in this case we are on Ubuntu 22.04, Jammy jellifish.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

It is possible to run the above command also to update an existing installation of OLLAMA.
Test it by running:

```bash
ollama run gemma3
```

which should pull the *llama3.2* model from the ollama repository and run it as a service in local (port 127.0.0.1:port_number).
Then, you can decide which OLLAMA model to use by downloading it. This is the output of *ollama list*:

```bash
NAME                  ID              SIZE      MODIFIED           
deepseek-r1:32b       edba8017331d    19 GB     About a minute ago    
deepseek-r1:latest    6995872bfe4c    5.2 GB    16 minutes ago        
gemma3:latest         a2af6cc3eb7f    3.3 GB    2 days ago            
gemma3:27b            a418f5838eaf    17 GB     2 days ago 
```

and with *ollama list*, *ollama ps*, *ollama rm* it is possible to list downloaded OLLAMA models, list the OLLAMA model currently running, and delete a downloaded OLLAMA model. 

Then to employ the OLLAMA model inside python code, we create a python virtualenv using *requirements.txt*. Hence, as usual, we run the following sequence of commands (assuming Python is already installed in your machine, here we are using Python 3.11.4 from Anaconda):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Now you should be able to call OLLAMA models also from the python code. Like 

```python
import OLLAMA

response = ollama.chat(model='gemma3', messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            }
    ])
print(response['message']['content'])
```

## ▶️ Run the System

Follow the preprocessing in the *data* folder to build the JSON messages that will build the stream of incoming messages:

```bash
cd data
python preprocessing.py --nest True
```

more instructions and infromation in *data/README.md*.
Then, the Schema Harmonization and JSON message transformation tasks can be perfromed by choosing an LLM from the downloaded OLLAMA suite. For example:

```python
cd ..
python main.py --model_name gemma3
```

or 

```python
cd ..
./run.sh
```

which will generate a folder inside the *results* directory with harmonized schemas after each JSON message from the stream, CSV files with initial structural/semantic similarity (their definition is in *utils/metrics*) between each pair of JSON messages in the stream, final harmonized schema and structural/semantic similarity between each JSON message in the stream, the final harmonized schema, and the transformed JSON messages to the harmonized schema. In the *results* folder it is possible to run a script for visualize structural/semantic similarity (namely *results/data_visualization.py*).

```python
cd results
python data_visualization.py -h
```

or 

```python
cd results
./run.sh
```