import argparse
import json
import os
import random

import torch
from datasets import load_dataset

from xchat.eval.mbpp.data import read_problems, write_jsonl
from xchat.eval.mbpp.evaluation import evaluate_functional_correctness
from xchat.eval.mbpp.prompts import getprompts
from xchat.eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model


def prepare_inference_arg(args, test_data):
    res = []
    for example in test_data:
        d = {}
        d["prompt"] = getprompts(args.chat_format, args.few_shot).get_prompt(example)
        d["reference"] = example["reference"]
        d["task_id"] = example["task_id"]
        res.append(d)
    return res


def prepare_data(args):
    if not os.path.exists(os.path.join(args.save_dir, "mbpp_inference_args.jsonl")):
        os.makedirs(args.save_dir, exist_ok=True)
        raw_dataset = load_dataset(path="mbpp", name=None, cache_dir=args.cache_dir)["test"]
        if len(raw_dataset) != 500:
            raise ValueError(f"Expected 500 test examples, got {len(raw_dataset)}")
        with open(os.path.join(args.save_dir, "mbpp_inference_args.jsonl"), "w") as f:
            for i, example in enumerate(raw_dataset):
                d__ = {
                    "task_id": i,
                    "prompt": example["text"],
                    "reference": "\n".join(example["test_list"]),
                }
                f.write(json.dumps(d__) + "\n")
    return os.path.join(args.save_dir, "mbpp_inference_args.jsonl")


def main(args):
    random.seed(42)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    data_file = prepare_data(args)

    test_data = list(read_problems(data_file).values())
    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    test_data = prepare_inference_arg(args, test_data)
    print("Number of examples:", len(test_data))

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            load_in_half=True,
            # device map is determined by the number of gpus available.
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 0
        test_data = sorted(
            test_data, key=lambda x: len(tokenizer(x["prompt"], return_tensors="pt").input_ids[0]), reverse=True
        )

        stop_sequences = ["\nassert", "\nprint"]
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        prompts = [example["prompt"] for example in test_data]

        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            assert args.greedy_decoding is True
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            samping_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                do_sample=not args.greedy_decoding,  # if only pass@1 is evaluated, we do greedy decoding.
                top_p=args.top_p,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
            assert len(samping_outputs) == len(prompts)
            outputs_per_sampling_iter.append(samping_outputs)
        outputs = []
        for i in range(len(prompts)):
            for j in range(args.unbiased_sampling_size_n):
                outputs.append(outputs_per_sampling_iter[j][i])
    else:
        instances = [
            {
                "id": examle["task_id"],
                "prompt": "Complete the following python function. Please only output the code for the completed function.\n\n\n"
                + prompt,
            }
            for examle, prompt in zip(test_data, prompts)
        ]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            output_path=os.path.join(args.save_dir, "openai_query_results.jsonl"),
            batch_size=args.eval_batch_size,
            top_p=0.95,
            temperature=args.temperature,
            n=args.unbiased_sampling_size_n,
        )
        outputs = []
        for result in results:
            for choice in result["response_metadata"]["choices"]:
                outputs.append(choice["message"]["content"])

    duplicate_test_data = [example for example in test_data for _ in range(args.unbiased_sampling_size_n)]
    assert len(duplicate_test_data) == len(outputs)

    def process(args, s):
        parse_seq = getprompts(args.chat_format, args.few_shot).get_parser_seq()
        return s.split(parse_seq)[0].strip()

    predictions = [
        {
            "task_id": example["task_id"],
            "prompt": example["prompt"],
            "completion": process(args, output.strip()),
            "reference": example["reference"],
        }
        for example, output in zip(duplicate_test_data, outputs)
    ]
    prediction_save_path = os.path.join(args.save_dir, "mbpp_predictions.jsonl")
    write_jsonl(prediction_save_path, predictions, append=True)

    pass_at_k_results = evaluate_functional_correctness(
        sample_file=prediction_save_path,
        k=args.eval_pass_at_ks,
        problems={example["task_id"]: example for example in test_data},
    )

    print(pass_at_k_results)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="Path to the data file.")
    parser.add_argument("--max_num_examples", type=int, default=None, help="Maximum number of examples to evaluate.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default=None, help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument("--data_dir", type=str, help="Directory to save the data.")
    parser.add_argument("--cache_dir", type=str, help="Directory to save the cache.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--max_new_tokens", type=int, required=True, help="Maximum number of tokens to generate.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--unbiased_sampling_size_n", type=int, default=1, help="Number of unbiased samples.")
    parser.add_argument("--few_shot", action="store_true", help="If given, we're evaluating 3-shot performance.")
    parser.add_argument("--greedy_decoding", action="store_true")
    parser.add_argument("--top_p", type=float)
    parser.add_argument(
        "--eval_pass_at_ks", nargs="+", type=int, default=[1], help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load mdodel in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--chat_format", type=str)
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
