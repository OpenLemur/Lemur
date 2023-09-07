import argparse
import glob
import json
import os
import random
import re

import evaluate
import torch
import tqdm

from xchat.eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model

exact_match = evaluate.load("exact_match")


@torch.no_grad()
def eval_hf_model(args, model, tokenizer, examples, task_prompt, save_path=None):
    targets = [example["target"] for example in examples]
    if save_path:
        fout = open(save_path, "w")

    prompts = []
    for example in examples:
        if args.chat_format == "tulu":
            prompt = "<|user|>\n" + task_prompt.strip() + "\n\nQ: " + example["input"] + "\n<|assistant|>\nA:"
        elif args.chat_format == "lemur":
            prompt = (
                "<|im_start|>user\n"
                + task_prompt.strip()
                + "\n\nQ: "
                + example["input"]
                + "<|im_end|>\n<|im_start|>assistant\nA:"
            )
        elif args.chat_format == "codellama-instruct":
            prompt = "[INST]" + task_prompt.strip() + "\n\nQ:" + example["input"] + "[/INST]" + "\nA:"
        else:
            prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA: "
        prompts.append(prompt)

    new_line_sequence = tokenizer.encode("\n", add_special_tokens=False)[-1]
    q_sequence = tokenizer.encode("Q: ", add_special_tokens=False)
    if args.chat_format == "tulu":
        stop_id_sequences = [
            [tokenizer.encode("<|assistant|>", add_special_tokens=False)[-1]],
            [new_line_sequence],
            [q_sequence],
        ]
    elif args.chat_format == "lemur":
        stop_id_sequences = [
            [tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]],
            [tokenizer.eos_token_id],
            [new_line_sequence],
            [q_sequence],
        ]
    elif args.chat_format == "codellama-instruct":
        stop_id_sequences = [
            [tokenizer.encode("[INST]", add_special_tokens=False)[-1]],
            [tokenizer.encode("</s>", add_special_tokens=False)[-1]],
            [new_line_sequence],
            [q_sequence],
        ]
    else:
        stop_id_sequences = [
            [new_line_sequence],
        ]

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        stop_id_sequences=stop_id_sequences,
    )

    predictions = []
    for example, output in zip(examples, outputs):
        example["raw_output"] = output

        # only keep the first part of the output - this is mainly for vanilla language models.
        output = output.strip().split("\n\n")[0].strip()

        # extract the first answer after `So the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        results = re.search(r"(.*?)\.", output)

        prediction = results.group(1).strip() if results else output.strip()

        if "(" in prediction and ")" in prediction and "[" not in prediction and "]" not in prediction:
            pattern = r"\(\w\)"
            match = re.search(pattern, prediction)
            prediction = f"{match.group(0)}" if match else prediction

        example["prediction"] = prediction
        predictions.append(prediction)
        if save_path:
            fout.write(json.dumps(example) + "\n")

    assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)[
        "exact_match"
    ]


def eval_openai_chat_engine(args, examples, task_prompt, save_path=None):
    targets = [example["target"] for example in examples]
    instances = []
    for i, example in enumerate(examples):
        prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
        instances.append(
            {
                "id": example["id"] if "id" in example else i,
                "prompt": prompt,
            }
        )

    if save_path:
        openai_result_save_path = os.path.join(
            os.path.dirname(save_path), os.path.basename(save_path).split(".")[0] + "_openai_results.jsonl"
        )

    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=openai_result_save_path if save_path else None,
    )

    outputs = [result["output"] for result in results]
    assert len(outputs) == len(targets), "number of predictions and targets are not the same."

    if save_path:
        fout = open(save_path, "w")

    predictions = []
    for example, output in zip(examples, outputs):
        example["raw_output"] = output
        # extract the first answer after `So the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        results = re.search(r"So the answer is (.*?)\.", output)
        prediction = results.group(1).strip() if results else output.strip()
        example["prediction"] = prediction
        predictions.append(prediction)
        if save_path:
            fout.write(json.dumps(example) + "\n")

    assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)[
        "exact_match"
    ]


def main(args):
    random.seed(args.seed)

    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    task_files = sorted(task_files)
    if args.task_start_idx is not None and args.task_end_idx is not None:
        task_files = task_files[args.task_start_idx : args.task_end_idx]
    print(task_files)
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file) as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    cot_prompt_files = sorted(cot_prompt_files)
    if args.task_start_idx is not None and args.task_end_idx is not None:
        cot_prompt_files = cot_prompt_files[args.task_start_idx : args.task_end_idx]
    print(cot_prompt_files)
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file) as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert (
                            "So the answer is" in prompt_field
                        ), f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(
        all_prompts.keys()
    ), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            load_in_half=True,
            gptq_model=args.gptq,
        )

    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        if args.special_task is not None:
            special_tasks = args.special_task.split(",")
            if task_name not in special_tasks:
                continue
        task_examples = all_tasks[task_name]
        prompt = all_prompts[task_name]
        if args.model_name_or_path:
            task_perf = eval_hf_model(
                args,
                model,
                tokenizer,
                task_examples,
                prompt,
                save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"),
            )
        else:
            task_perf = eval_openai_chat_engine(
                args, task_examples, prompt, save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl")
            )
        performance[task_name] = task_perf
        print(f"Task {task_name} - EM: {task_perf}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bbh")
    parser.add_argument("--save_dir", type=str, default="results/bbh")
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
    parser.add_argument(
        "--no_cot", action="store_true", help="if specified, chain of thoughts will be removed from the prompts."
    )
    parser.add_argument(
        "--max_num_examples_per_task", type=int, default=None, help="maximum number of examples to evaluate per task."
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
        choices=["tulu", "lemur", "codellama-instruct"],
        help="If given, the prompt will be encoded as a chat format with the roles in prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--task_start_idx",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--task_end_idx",
        type=int,
        default=None,
    )
    parser.add_argument("--special_task", type=str, default=None)

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
