# An AI Driven Harmonized Data Ingestion: Vision For Seamless Cross-Domain Data Integration

## Setup

Install OLLAMA on your laptop from their official [website](https://ollama.com/download), more infromations can also be found at their GitHub [repo](https://github.com/ollama/ollama). Choose your OS, in this case we are on Ubuntu 22.04, Jammy jellifish.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Test it by running:

```bash
ollama run llama3.2
```

which should pull the *llama3.2* model from the ollama repository and run it as a service in local (port 127.0.0.1:port_number).
Then, you can decide which OLLAMA model to use by downloading it. For example to get *mistral*:

```bash
ollama pull mistral
```

we are going to test different of them which should be downloaded before using them in the code (not sure whether the download will occur automatically if you try to use a model that has not been pulled previously). We can also download the Qwen series model:

```bash
ollama pull qwen2.5
```

and with *ollama list* and *ollama ps* it is possible to see which OLLAMA models have been downloaded and which ones are currently running.

Then to use it inside python code, we create a python virtualenv using *requirements.txt*. Hence, as usual, we run the following sequence of commands (assuming you already installed Python in your machine, here we are using Python 3.11.4 from Anaconda):

```bash
python -m venv venv # venv is the name of the python virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

Now you should be able to call OLLAMA models also from the python code. Like 

```python
import OLLAMA

response = ollama.chat(model='mistral', messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            }
    ])
print(response['message']['content'])
```

> - With *mistral*, you should get: *The sky appears blue because of a phenomenon called Rayleigh scattering. As sunlight passes through Earth's atmosphere, it interacts with molecules and tiny particles in the air, like dust and water droplets. Blue light has shorter wavelengths and gets scattered more than other colors as it travels due to the way its particles are arranged. This scattered blue light is what we see when looking up at the sky. At sunrise or sunset, you may observe the sky in shades of red, orange, pink, and purple - this is due to a different type of scattering called Raman scattering and the way the sunlight interacts with air molecules at specific angles.*

> - With *llama3.2*: *The sky appears blue because of a phenomenon called scattering, which occurs when sunlight interacts with the tiny molecules of gases in the Earth's atmosphere.

Here's what happens:

1. Sunlight enters the Earth's atmosphere and contains all the colors of the visible spectrum (red, orange, yellow, green, blue, indigo, and violet).
2. When sunlight encounters a gas molecule, such as nitrogen or oxygen, it scatters in all directions.
3. The shorter (blue) wavelengths are scattered more than the longer (red) wavelengths because they interact more strongly with the smaller gas molecules.
4. As a result, the blue light is dispersed throughout the atmosphere, giving the sky its blue appearance.

This effect is known as Rayleigh scattering, named after the British physicist Lord Rayleigh, who first described it in the late 19th century. It's also responsible for the colorful sunsets we see when sunlight passes through more particles in the atmosphere.

So, to summarize, the sky appears blue because of the scattering of sunlight by tiny gas molecules in the Earth's atmosphere, which favors shorter wavelengths like blue over longer wavelengths like red.

Do you have any other questions about this topic?*

> - With *Qwen*: *The sky appears blue because of the way Earth's atmosphere scatters sunlight in all directions. Sunlight is made up of different colors that correspond to different wavelengths. When sunlight enters the Earth's atmosphere, the shorter wavelength colors (blue and violet) are scattered by gases and particles in the air much more than the longer wavelength colors (red, orange, and yellow). 

However, our eyes are more sensitive to blue light, and there is a lot less natural violet light, so we predominantly perceive the sky as being blue. At sunrise or sunset, you might see reds and oranges because by this time, most of the shorter wavelengths have been scattered out, leaving the longer wavelengths to reach your eyes directly.*

## Run the experiment

```python
python main.py
```
