class HumanEvalPromptBase(object):
    def __init__(self):
        pass
    
    def getPrompt(self, prompt):
        return prompt
        
class ChatMLPrompt(HumanEvalPromptBase):
    def __init__(self):
        super().__init__()
    
    def getPrompt(self, prompt):
        return "<|im_start|>user\n" + \
            "Complete the following python function.\n\n\n" + \
            prompt + \
            "<|im_end|>\n<|im_start|>assistant\n" + \
            "Here is the completed function:\n\n\n" + \
            prompt + \
            "<|im_end|>\n" 
    
class LLaMA2ChatPrompt(HumanEvalPromptBase):
    def __init__(self, prompt):
        super().__init__(prompt)
    
    def getPrompt(self, prompt):
        return  "[INST] <<SYS>>\n" \
        + "Below is an instruction that describes a task. Write a response that appropriately completes the request." \
        + "\n<</SYS>>\n\n" \
        + "Complete the following python function.\n\n\n" \
        + prompt \
        + "[/INST]" \
        + "Here is the completed function:\n\n\n" \
        + prompt


class VicunaPrompt(HumanEvalPromptBase):
    def __init__(self, prompt):
        super().__init__(prompt)
    
    def getPrompt(self, prompt):
        system_message = """Below is an instruction that describes a task. Write a response that appropriately completes the request."""
        return "USER: " + system_message + "Complete the following python function.\n\n\n" + prompt + "\nASSISTANT: " + "Here is the completed function:\n\n\n" + prompt

class TuluPrompt(HumanEvalPromptBase):
    def __init__(self, prompt):
        super().__init__(prompt)
    
    def getPrompt(self, prompt):
        return "<|user|>\n" \
            + "Complete the following python function.\n\n\n" \
            + prompt \
            + "\n<|assistant|>\n" \
            + "Here is the completed function:\n\n\n" \
            + prompt

def getprompts(chat_format):
    if chat_format in ["base"]:
        return HumanEvalPromptBase()
    if chat_format in ["chatml"]:
        return ChatMLPrompt()
    elif chat_format in ["tulu"]:
        return TuluPrompt()
    elif chat_format in ["llama2chat"]:
        return LLaMA2ChatPrompt()
    elif chat_format in ["vicuna"]:
        return VicunaPrompt()
    raise NotImplementedError
