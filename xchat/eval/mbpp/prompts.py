from xchat.eval.mbpp.examplars import EXAMPLARS


class MbppPromptBase:
    def __init__(self, instuction, end_seq, is_few_shot) -> None:
        self.instruction = instuction
        self.end_seq = end_seq
        self.is_few_shot = is_few_shot

    def get_prompt(self, example):
        prompt = ""
        if self.is_few_shot:
            for d in EXAMPLARS:
                prompt += (
                    self.instruction.format(instruction=d["text"], test_cases="\n".join(d["test_list"]), code=d["code"])
                    + self.end_seq
                )
        prompt += self.instruction.format(
            instruction=example["prompt"].strip(), test_cases=example["reference"], code=""
        )
        return prompt.strip()

    def get_parser_seq(self):
        return self.end_seq


class MbppPromptChat(MbppPromptBase):
    def __init__(self, instuction, end_seq, is_few_shot) -> None:
        super().__init__(instuction, end_seq, is_few_shot)

    def get_prompt(self, example):
        prompt = "[INST]"
        if self.is_few_shot:
            for d in EXAMPLARS:
                prompt += (
                    self.instruction.format(instruction=d["text"], test_cases="\n".join(d["test_list"]), code=d["code"])
                    + self.end_seq
                )
        prompt += self.instruction.format(
            instruction=example["prompt"].strip(), test_cases=example["reference"], code=""
        )
        prompt = prompt.strip()
        assert prompt.endswith("[PYTHON]")
        prompt = prompt[:-8]
        prompt += "[/INST]\n[PYTHON]\n"
        return prompt.strip()


BASE_INSTRUCTION = """
You are an expert Python programmer, and here is your task: {instruction} Your code should pass these tests:\n\n{test_cases}
[BEGIN]
{code}"""


class BasePrompt(MbppPromptBase):
    def __init__(self, is_few_shot) -> None:
        super().__init__(instuction=BASE_INSTRUCTION, end_seq="\n[DONE]", is_few_shot=is_few_shot)


INSTRUCTION = """
You are an expert Python programmer, and here is your task: {instruction}
Your code should pass these tests:\n\n{test_cases}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.
[PYTHON]
{code}"""


class CodeLLaMAPrompt(MbppPromptBase):
    def __init__(self, is_few_shot) -> None:
        super().__init__(instuction=INSTRUCTION, end_seq="\n[/PYTHON]", is_few_shot=is_few_shot)


class LLaMA2ChatPrompt(MbppPromptChat):
    def __init__(self, is_few_shot) -> None:
        super().__init__(instuction=INSTRUCTION, end_seq="\n[/PYTHON]", is_few_shot=is_few_shot)


LEMUR_INSTRUCTION = """
<|im_start|>user
You are an expert Python programmer, and here is your task: {instruction}
Your code should pass these tests:\n\n{test_cases}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag.
<|im_end|>
<|im_start|>assistant
[PYTHON]
{code}"""


class LemurPrompt(MbppPromptBase):
    def __init__(self, is_few_shot) -> None:
        super().__init__(instuction=LEMUR_INSTRUCTION, end_seq="\n[/PYTHON]", is_few_shot=is_few_shot)


def getprompts(chat_format, is_few_shot):
    if chat_format in ["base"]:
        return BasePrompt(is_few_shot)
    elif chat_format in ["lemur"]:
        return LemurPrompt(is_few_shot)
    elif chat_format in ["codellama", "wizardcoder"]:
        return CodeLLaMAPrompt(is_few_shot)
    elif chat_format in ["llama2chat"]:
        return LLaMA2ChatPrompt(is_few_shot)
    raise NotImplementedError
