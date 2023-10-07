# Lemur: Open Foundation Models for Language Agents

<p align="center">
  <img src="https://huggingface.co/datasets/OpenLemur/assets/resolve/main/lemur_icon.png" width="300" height="300" alt="Lemur">
</p>
   <a href="https://huggingface.co/OpenLemur" target="_blank">
      <img alt="Models" src="https://img.shields.io/badge/🤗-Models-blue" />
   </a>
   <a href="https://xlang.ai/blog/openlemur" target="_blank">
      <img alt="Blog" src="https://img.shields.io/badge/📖-Blog-red" />
   </a>
  <a href="https://xlang.ai" target="_blank">
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper(Coming soon)-purple" />
   </a>
   <a href="https://github.com/OpenLemur/lemur" target="_blank">
      <img alt="Stars" src="https://img.shields.io/github/stars/OpenLemur/lemur?style=social" />
   </a>
   <a href="https://github.com/OpenLemur/lemur/issues" target="_blank">
      <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/OpenLemur/lemur" />
   </a>
   <a href="https://twitter.com/XLangAI" target="_blank">
      <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/XLangAI" />
   </a>
   <a href="https://join.slack.com/t/xlanggroup/shared_invite/zt-20zb8hxas-eKSGJrbzHiPmrADCDX3_rQ" target="_blank">
      <img alt="Join Slack" src="https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp" />
   </a>
   <a href="https://discord.gg/ncjujmva" target="_blank">
      <img alt="Discord" src="https://dcbadge.vercel.app/api/server/ncjujmva?compact=true&style=flat" />
   </a>

Open large language models (LLMs) have traditionally been tailored for either textual or code-related tasks, with limited ability to effectively balance both. However, many complex language applications, particularly language model agents, demand systems with a multifaceted skill set encompassing understanding, reasoning, planning, coding, and context grounding.

In this work, we introduce **Lemur-70B-v1** and **Lemur-70B-chat-v1**, the state-of-the-art open pretrained and supervised fine-tuned large language models balancing text and code intelligence.

<div align="center">
  <img src="https://huggingface.co/datasets/OpenLemur/assets/resolve/main/lemur_performance.png">
</div>

This release includes model weights and starting code for using Lemur models, and we will continue to update more models and code.

This repository is a minimal example to load Lemur models, run inference, and be initialized for further finetuning. Check [huggingface](https://huggingface.co/OpenLemur) for a more detailed usage recipe.


## 🔥 News
* **[23 August, 2023]:** 🎉 We release the weights of [`OpenLemur/lemur-70b-v1`](https://huggingface.co/OpenLemur/lemur-70b-v1), and [`OpenLemur/lemur-70b-chat-v1`](https://huggingface.co/OpenLemur/lemur-70b-chat-v1)! Check it out in [HuggingFace Hub](https://huggingface.co/OpenLemur).


## Table of Contents
- [Quickstart](#quickstart)
  - [Setup](#setup)
  - [Models](#models)
  - [Inference](#inference)
    - [Pretrained Model](#pretrained-model)
    - [Supervised Fine-tuned Model](#supervised-fine-tuned-model)
- [Acknowledgements](#acknowledgements)

## Quickstart

### Setup
First, we have to install all the libraries listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Models

Model cards are published on the HuggingFace Hub:

* [OpenLemur/lemur-70b-v1](https://huggingface.co/OpenLemur/lemur-70b-v1)
* [OpenLemur/lemur-70b-chat-v1](https://huggingface.co/OpenLemur/lemur-70b-chat-v1)


### Inference

You can do inference like this:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("OpenLemur/lemur-70b-v1")
model = AutoModelForCausalLM.from_pretrained("OpenLemur/lemur-70b-v1", device_map="auto", load_in_8bit=True)

prompt = "Your prompt here"
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

#### Pretrained Model

The model is initialized from LLaMa-2 70B and further trained on ~100B text and code data (more details coming soon!). They should be prompted so that the expected answer is the natural continuation of the prompt.

Here is a simple example of using our pretrained model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("OpenLemur/lemur-70b-v1")
model = AutoModelForCausalLM.from_pretrained("OpenLemur/lemur-70b-v1", device_map="auto", load_in_8bit=True)

# Text Generation Example
prompt = "The world is "
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Code Generation Example
prompt = """
def factorial(n):
    if n == 0:
        return 1
"""
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=200, num_return_sequences=1)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
```


### Supervised Fine-tuned Model

The model is initialized from `lemur-70b-v1` and continues trained on supervised fine-tuning data.

Here is a simple example to use our supervised fine-tuned model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("OpenLemur/lemur-70b-chat-v1")
model = AutoModelForCausalLM.from_pretrained("OpenLemur/lemur-70b-chat-v1", device_map="auto", load_in_8bit=True)

# Text Generation Example
prompt = """<|im_start|>system
You are a helpful, respectful, and honest assistant.
<|im_end|>
<|im_start|>user
What's a lemur's favorite fruit?<|im_end|>
<|im_start|>assistant
"""
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Code Generation Example
prompt = """<|im_start|>system
Below is an instruction that describes a task. Write a response that appropriately completes the request.
<|im_end|>
<|im_start|>user
Write a Python function to merge two sorted lists into one sorted list without using any built-in sort functions.<|im_end|>
<|im_start|>assistant
"""
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=200, num_return_sequences=1)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
```

## Evaluation
We build the evaluation suite based on [open-instruct](https://github.com/allenai/open-instruct). We will keep updating more tasks and models.

### Tasks Table

|                               | MMLU | BBH | GSM8K | HumanEval | MBPP | DS-1000 | MultiPL-E |
|-------------------------------|------|-----|-------|-----------|------|---------|-----------|
| Llama-2-{7,13,70}b            | ✅    | ✅   | ✅     | ✅         | ✅    | 🚧       | 🚧         |
| CodeLlama-{7,13,34}b          | ✅    | ✅   | ✅     | ✅         | ✅    | 🚧       | 🚧         |
| Lemur-70b-v1                  | ✅    | ✅   | ✅     | ✅         | ✅    | 🚧       | 🚧         |
| Llama-2-{7,13,70}b-chat       | ✅    |     | ✅     | ✅         | ✅    | 🚧       | 🚧         |
| CodeLlama-{7,13,34}b-Instruct | ✅    |     | ✅     | ✅         | ✅    | 🚧       | 🚧         |
| Lemur-70b-chat-v1             | ✅    |     | ✅     | ✅         | ✅    | 🚧       | 🚧         |

### Setup
```bash
conda create -n xchat python=3.10
conda activate xchat
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```
Then, install the `xchat` package:

```bash
git clone git@github.com:OpenLemur/Lemur.git
cd Lemur
pip install -e .
```

### Run Evaluation
Please find the evaluation scripts in `scripts/eval`. For example, to run the GSM8K evaluation, update the arguments in `scripts/eval/gsm.sh` and run:

```bash
bash scripts/eval/gsm.sh
```

## Acknowledgements

The Lemur project is an open collaborative research effort between [XLang Lab](https://www.xlang.ai/) and [Salesforce Research](https://www.salesforceairesearch.com/). We thank the following institutions for their gift support:

<div align="center">

<img src="assets/transparent.png" width="20" style="pointer-events: none;">

<a href="https://www.salesforceairesearch.com/">
    <img src="assets/salesforce.webp" alt="Salesforce Research" height = 30/>
</a>

<img src="assets/transparent.png" width="20" style="pointer-events: none;">

<a href="https://research.google/">
    <img src="assets/google_research.svg" alt="Google Research" height = 30/>
</a>

<img src="assets/transparent.png" width="25" style="pointer-events: none;">

<a href="https://www.amazon.science/" style="display: inline-block; margin-bottom: -100px;">
    <img src="assets/amazon.svg" alt="Amazon AWS" height = 20 />
</a>


</div>
