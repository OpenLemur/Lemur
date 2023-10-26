# [Lemur: Open Foundation Models for Language Agents](https://arxiv.org/abs/2310.06830)

<p align="center">
  <img src="https://i.imgur.com/Tga8kHW.jpeg" alt="Lemur">
</p>
   <a href="https://huggingface.co/OpenLemur" target="_blank">
      <img alt="Models" src="https://img.shields.io/badge/🤗-Models-blue" />
   </a>
   <a href="https://xlang.ai/blog/openlemur" target="_blank">
      <img alt="Blog" src="https://img.shields.io/badge/📖-Blog-red" />
   </a>
  <a href="https://arxiv.org/abs/2310.06830" target="_blank">
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" />
   </a>
   <a href="https://github.com/OpenLemur/lemur" target="_blank">
      <img alt="Stars" src="https://img.shields.io/github/stars/OpenLemur/lemur?style=social" />
   </a>
   <a href="https://github.com/OpenLemur/lemur/issues" target="_blank">
      <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/OpenLemur/lemur" />
   </a>
   <a href="https://twitter.com/XLangNLP" target="_blank">
      <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/XLANG NLP Lab" />
   </a>
   <a href="https://join.slack.com/t/xlanggroup/shared_invite/zt-20zb8hxas-eKSGJrbzHiPmrADCDX3_rQ" target="_blank">
      <img alt="Join Slack" src="https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp" />
   </a>
   <a href="https://discord.gg/4Gnw7eTEZR" target="_blank">
      <img alt="Discord" src="https://dcbadge.vercel.app/api/server/4Gnw7eTEZR?compact=true&style=flat" />
   </a>

Lemur is an openly accessible language model optimized for both natural language and coding capabilities to serve as the backbone of versatile language agents.
As language models continue to evolve from conversational chatbots to functional agents that can act in the real world, they need both strong language understanding and the ability to execute actions. Lemur balances natural language and coding skills to enable agents to follow instructions, reason for tasks, and take grounded actions.

<div align="center">
  <img src="./assets/interface.png">
</div>

Please refer to our paper and code for more details:
- [[Paper](https://arxiv.org/abs/2310.06830)] Lemur: Harmonizing Natural Language and Code for Language Agents
- [[Blog](https://www.xlang.ai/blog/openlemur)] Introducing Lemur: Open Foundation Models for Language Agents


## 🔥 News
* **[18 October, 2023]:** 🎉 We open-sourced [code for OpenAgents](https://github.com/xlang-ai/OpenAgents): An Open Platform for Language Agents in the Wild. 
* **[11 October, 2023]:** 🎉 We released the research paper and codebase. We will continue updating this repository.
* **[23 August, 2023]:** 🎉 We released the weights of [`OpenLemur/lemur-70b-v1`](https://huggingface.co/OpenLemur/lemur-70b-v1), and [`OpenLemur/lemur-70b-chat-v1`](https://huggingface.co/OpenLemur/lemur-70b-chat-v1)! Check it out in [HuggingFace Hub](https://huggingface.co/OpenLemur).

## Models
We released our models on the HuggingFace Hub:
* [OpenLemur/lemur-70b-v1](https://huggingface.co/OpenLemur/lemur-70b-v1)
* [OpenLemur/lemur-70b-chat-v1](https://huggingface.co/OpenLemur/lemur-70b-chat-v1)

## Table of Contents
- [Lemur: Open Foundation Models for Language Agents](#lemur-open-foundation-models-for-language-agents)
  - [🔥 News](#-news)
  - [Models](#models)
  - [Table of Contents](#table-of-contents)
  - [Why Lemur?](#why-lemur)
  - [Quickstart](#quickstart)
    - [Setup](#setup)
    - [Lemur-70B](#lemur-70b)
    - [Lemur-70B-Chat](#lemur-70b-chat)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Foundational Abilities](#foundational-abilities)
    - [Interactive Agent Skills](#interactive-agent-skills)
      - [Deploy](#deploy)
      - [MINT](#mint)
      - [WebArena](#webarena)
      - [InterCode](#intercode)
      - [RobotCodeGen](#robotcodegen)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)



## Why Lemur?
Most existing open-source models specialize in either natural language or code. Lemur combines both strengths by:

- Pretraining Llama-2-70B on a 90B token corpus with the 10:1 ratio of code to text and obtaining Lemur-70B-v1
- Instruction tuning Lemur-70B-v1 on 300K examples covering both text and code and obtaining Lemur-70B-Chat-v1

This two-stage training produces state-of-the-art performance averaged across diverse language and coding benchmarks, surpassing other available open-source models and narrowing the gap between open-source and commercial models on agent abilities.

## Quickstart

### Setup
First, we have to install all the libraries listed in `requirements.txt`

```bash
conda create -n xchat python=3.10
conda activate xchat
conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```
Then, install the xchat package:
```bash
git clone git@github.com:OpenLemur/Lemur.git
cd Lemur
pip install -e .
```

### Lemur-70B
For the base model lemur-70b-v1, you can use it in this way:

<details>
<summary>Click me</summary>
<p>

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

</p>

</details>


### Lemur-70B-Chat
We instruction-finetune lemur-70b-v1 model with ChatML format to obtain lemur-70b-chat-v1. You can use lemur-70b-chat-v1 in this way:

<details>
<summary>Click me</summary>
<p>

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

</p>

</details>

## Training

<div align="center">
  <img src="./assets/training.png">
</div>


## Evaluation
We evaluated Lemur across:
- 8 language and code datasets like MMLU, BBH, GSM8K, HumanEval, and Spider to validate balanced capabilities
- 13 interactive agent datasets to test skills like tool usage, adapting to feedback from environments or humans, and exploring partially observable digital or physical environments.

<div align="center">
  <img src="./assets/agent-scenarios.png">
</div>

<div align="center">
  <img src="./assets/overall-perform.png">
</div>

### Foundational Abilities
We build the evaluation suite based on [open-instruct](https://github.com/allenai/open-instruct). We will keep updating more tasks and models.

Currently, we support the following tasks:
- [✅] [MMLU](./scripts/eval/mmlu.sh)
- [✅] [BBH](./scripts/eval/bbh.sh)
- [✅] [GSM8K](./scripts/eval/gsm8k.sh)
- [✅] [HumanEval](./scripts/eval/human_eval.sh)
- [✅] [MBPP](./scripts/eval/mbpp.sh)
- [🚧] [Spider]()
- [🚧] [MultiPL-E]()
- [🚧] [DS-1000]()
- [🚧] ...

### Interactive Agent Skills
We use the evaluation frameworks provided by [MINT](https://github.com/xingyaoww/mint-bench), [InterCode](https://github.com/princeton-nlp/intercode), and [WebArena](https://github.com/web-arena-x/webarena) to evaluate interactive agent skills.

#### Deploy
We use vLLM to serve the Lemur model. However, the official FastChat codebase does not yet support Lemur-Chat. Therefore, we provide a docker to serve vLLM for Lemur. Please refer to [vllm_lemur.sh](./scripts/deploy/vllm_lemur.sh) for more detailed information.

```bash
bash scripts/deploy/vllm_lemur.sh
```

#### MINT
We [fork MINT](https://github.com/OpenLemur/mint-bench) codebase to share the configs we used. Please refer to [this config folder](https://github.com/OpenLemur/mint-bench/tree/main/configs) for more details. Please run vllm with [`vllm_lemur.sh`](./scripts/deploy/vllm_lemur.sh) script.

#### WebArena
We [fork WebArena](https://github.com/OpenLemur/webarena) codebase to enable vLLM evaluation. To run the evaluation on WebArena, please refer to our [forked WebArena codebase](https://github.com/OpenLemur/webarena).

#### InterCode
🚧 Working in progress

#### RobotCodeGen
🚧 Working in progress

## Citation
If you find our work helpful, please cite us:
```
@misc{xu2023lemur,
      title={Lemur: Harmonizing Natural Language and Code for Language Agents}, 
      author={Yiheng Xu and Hongjin Su and Chen Xing and Boyu Mi and Qian Liu and Weijia Shi and Binyuan Hui and Fan Zhou and Yitao Liu and Tianbao Xie and Zhoujun Cheng and Siheng Zhao and Lingpeng Kong and Bailin Wang and Caiming Xiong and Tao Yu},
      year={2023},
      eprint={2310.06830},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
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
