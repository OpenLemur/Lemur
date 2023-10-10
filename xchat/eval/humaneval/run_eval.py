import argparse
import json
import os
import random

import torch

from xchat.eval.humaneval.prompts import getprompts
from xchat.eval.mbpp.data import read_problems, write_jsonl
from xchat.eval.mbpp.evaluation import evaluate_functional_correctness
from xchat.eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model


def main(args):
    random.seed(42)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    data_file = args.data_file
    test_data = list(read_problems(data_file).values())

    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("Number of examples:", len(test_data))
    promptclass = getprompts(args.chat_format)
    prompts = [promptclass.getPrompt(example["prompt"]) for example in test_data]

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

        # these stop sequences are those mentioned in the codex paper.
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
        # Because many tokenizers will treat the word after space differently from the original word alone,
        # to be consistent, we add a space before tokenization and remove it after tokenization.
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        print("stop sequences:")
        print(stop_sequences)
        if args.chat_format == "vicuna":
            # Vicuna uses a different stop sequence otherwise meaning tokens may be cut off.
            stop_sequences = [[tokenizer.eos_token_id]]
        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            samping_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                top_p=args.top_p,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs_per_sampling_iter.append(samping_outputs)
        # regroup the outputs to match the number of test data.
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

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [example for example in test_data for _ in range(args.unbiased_sampling_size_n)]
    predictions = []
    for _, example, output in zip(prompts, duplicate_test_data, outputs):
        output_lines = output.split("\n")
        idx = len(output_lines) - 1
        while idx >= 0 and not output_lines[idx].strip().startswith("return"):
            idx -= 1
        output_lines = output_lines[: idx + 1]
        processed_output = "\n".join(output_lines) + "\n"
        predictions.append(
            {
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "completion": processed_output,
                "raw_response": output,
            }
        )

        prediction_save_path = os.path.join(args.save_dir, "humaneval_predictions.jsonl")

    write_jsonl(prediction_save_path, predictions)
    pass_at_k_results = evaluate_functional_correctness(
        task="humaneval",
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
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k.",
    )
    parser.add_argument(
        "--eval_pass_at_ks", nargs="+", type=int, default=[1], help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n",
        type=int,
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. ",
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument(
        "--chat_format",
        type=str,
        default=None,
        choices=["tulu", "chatml", "llama-2-chat", "vicuna", "base"],
        help="if given, we will use the chat format to generate the predictions.",
    )
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    assert args.unbiased_sampling_size_n >= max(
        args.eval_pass_at_ks
    ), "n should be larger than the largest k in eval_pass_at_ks."
    main(args)
