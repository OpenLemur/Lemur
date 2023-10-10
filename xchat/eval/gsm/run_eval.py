# Copied and modified from https://github.com/allenai/open-instruct

import argparse
import json
import os
import random
import re

import evaluate

from xchat.eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS
from xchat.eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model

exact_match = evaluate.load("exact_match")


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({"question": example["question"], "answer": example["answer"].split("####")[1].strip()})

    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    global GSM_EXAMPLARS
    if args.n_shot:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in GSM_EXAMPLARS:
            if args.no_cot:
                demonstrations.append("Question: " + example["question"] + "\n" + "Answer: " + example["short_answer"])
            else:
                demonstrations.append("Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"])
        prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
    else:
        prompt_prefix = "Answer the following question.\n\n"

    prompts = []
    for example in test_data:
        if args.chat_format == "tulu":
            prompt = (
                "<|user|>\n"
                + prompt_prefix
                + "Question: "
                + example["question"].strip()
                + "\n<|assistant|>\n"
                + "Answer:"
            )
        elif args.chat_format == "lemur":
            prompt = (
                "<|im_start|>user\n"
                + prompt_prefix
                + "Question: "
                + example["question"].strip()
                + "<|im_end|>\n<|im_start|>assistant\n"
                + "Answer:"
            )
        elif args.chat_format == "wizardcoder":
            prompt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{prompt_prefix}"
                "Question: "
                f"{example['question'].strip()}"
                "\n### Response:\n"
                "Answer:"
            )
        elif args.chat_format == "vicuna":
            prompt = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                "USER: " + prompt_prefix + "Question: " + example["question"].strip() + "\nASSISTANT: " + "Answer:"
            )
        elif args.chat_format == "codellama-instruct":
            prompt = "[INST] " + prompt_prefix + "Question: " + example["question"].strip() + "[/INST]" + "Answer:"
        elif args.chat_format == "llama-2-chat":
            system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
            prompt = (
                "[INST] <<SYS>>\n"
                f"{system_message}"
                "\n<</SYS>>\n\n"
                f"{prompt_prefix}"
                "Question: "
                f"{example['question'].strip()}"
                "[/INST]"
                "Answer:"
            )
        else:
            prompt = prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:"
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            load_in_half=True,
            gptq_model=args.gptq,
        )

        if args.chat_format == "tulu":
            stop_id_sequences = [[tokenizer.encode("<|assistant|>", add_special_tokens=False)[-1]]]
        elif args.chat_format == "lemur":
            stop_id_sequences = [
                [tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]],
                [tokenizer.eos_token_id],
            ]
        elif args.chat_format == "vicuna" or args.chat_format == "llama-2-chat":
            stop_id_sequences = [[tokenizer.encode("</s>", add_special_tokens=False)[-1]]]
        elif args.chat_format == "codellama-instruct":
            stop_id_sequences = [
                [tokenizer.encode("[INST]", add_special_tokens=False)[-1]],
                [tokenizer.encode("</s>", add_special_tokens=False)[-1]],
            ]
        else:
            # get the last token because the tokenizer may add space tokens at the start.
            stop_id_sequences = [[tokenizer.encode("\n", add_special_tokens=False)[-1]]]
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            stop_id_sequences=stop_id_sequences,
        )
    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, "openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    raw_predictions = [
        {"question": example["question"], "answer": example["answer"], "model_output": output, "prediction": pred}
        for example, output, pred in zip(test_data, outputs, predictions)
    ]

    with open(os.path.join(args.save_dir, "raw_predictions.jsonl"), "w") as fout:
        for raw_prediction in raw_predictions:
            fout.write(json.dumps(raw_prediction) + "\n")

    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = exact_match.compute(
        predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True
    )["exact_match"]
    print(f"Exact match : {em_score}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({"exact_match": em_score}, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument("--n_shot", type=int, default=8, help="max number of examples to use for demonstration.")
    parser.add_argument(
        "--no_cot", action="store_true", help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument(
        "--chat_format",
        type=str,
        default=None,
        choices=["tulu", "lemur", "wizardcoder", "vicuna", "llama-2-chat", "codellama-instruct"],
        help="if given, we will use the chat format to generate the predictions.",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
