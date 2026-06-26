import os
import re
import json
import time
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)

    parser.add_argument("--judge_model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://mmia1-8293-resource.openai.azure.com/openai/v1/",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    parser.add_argument("--judge_max_tokens", type=int, default=512)
    parser.add_argument("--expected_final_max_tokens", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)

    return parser.parse_args()


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    data = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def save_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def get_query(item):
    return (
        item.get("original_query")
        or item.get("problem")
        or item.get("prompt")
        or item.get("query")
        or item.get("user_query")
        or ""
    )


def get_reference(item):
    return (
        item.get("reference_answer")
        or item.get("answer")
        or item.get("solution")
        or ""
    )


def get_response(item):
    return (
        item.get("final_response")
        or item.get("response")
        or item.get("model_response")
        or item.get("raw_response")
        or ""
    )


def get_raw_response(item):
    return item.get("raw_response") or get_response(item)


def get_reasoning_text(item):
    return (
        item.get("reasoning_text")
        or item.get("token_report", {}).get("reasoning_text")
        or ""
    )


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def detect_final_token_limit_hit(item, expected_final_max_tokens=None):
    candidates = [
        item.get("hit_max_tokens"),
        safe_get(item, ["token_report", "hit_max_tokens"]),
        safe_get(item, ["final_token_report", "hit_max_tokens"]),
        safe_get(item, ["final_usage", "hit_max_tokens"]),
    ]

    for c in candidates:
        if c is True:
            return True

    finish_reason_candidates = [
        item.get("finish_reason"),
        item.get("final_finish_reason"),
        safe_get(item, ["final_usage", "finish_reason"]),
        safe_get(item, ["token_report", "finish_reason"]),
    ]

    for fr in finish_reason_candidates:
        if isinstance(fr, str) and fr.lower() in {"length", "max_tokens", "token_limit"}:
            return True

    output_tokens = (
        item.get("output_tokens")
        or safe_get(item, ["token_report", "output_tokens"])
        or safe_get(item, ["final_usage", "output_tokens"])
        or safe_get(item, ["usage", "completion_tokens"])
    )

    if expected_final_max_tokens is not None and isinstance(output_tokens, (int, float)):
        if output_tokens >= expected_final_max_tokens:
            return True

    response = get_response(item).strip()
    raw = get_raw_response(item).strip().lower()

    truncation_markers = [
        "maximum context length",
        "max tokens",
        "token limit",
        "continued in next",
        "answer is incomplete",
    ]

    if any(m in raw for m in truncation_markers):
        return True

    if response.endswith(("...", "\\", "\\[", "\\(", ",", ";", ":")):
        return True

    return False


def is_content_filter_error(error_msg):
    low = error_msg.lower()
    return (
        "content_filter" in low
        or "responsibleaipolicyviolation" in low
        or "filtered due to the prompt" in low
    )


def extract_json_object(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def judge_answer_and_reasoning(
    client,
    model,
    query,
    reference,
    response,
    reasoning_text,
    raw_response,
    final_token_limit_hit,
    judge_max_tokens,
):
    prompt = f"""
You are a strict math correctness judge and reasoning-efficiency judge.

Evaluate the MODEL RESPONSE for the ORIGINAL QUERY using the REFERENCE ANSWER.

You must judge two things:

A) Correctness:
- true if the final answer is mathematically correct and consistent.
- false if the answer is wrong, incomplete, truncated, unclear, contradicted, or missing.

B) Reasoning efficiency:
Choose exactly one:
- "over_reasoning": The response uses excessive/redundant reasoning, repeats the same solution, includes unnecessary derivations, unnecessary tool-based explanation, or spends much more reasoning than needed for a simple problem, even if the final answer is correct.
- "under_reasoning": The response gives too little reasoning for the problem, skips necessary steps, makes unsupported jumps, is incomplete, or token-limit truncation prevents a complete solution.
- "adequate_reasoning": The reasoning is appropriate for the problem complexity.

Important rules:
1. If the model response reaches the correct final answer after long reasoning, correctness is true, but reasoning may still be over_reasoning.
2. If token limit was hit and the final answer is incomplete or unclear, correctness is false and reasoning is usually under_reasoning.
3. If token limit was hit but the final answer is still clearly correct and complete, correctness can be true.
4. Ignore LaTeX formatting differences.
5. Return only valid JSON.

Return this exact JSON schema:
{{
  "eval_result": true,
  "answer_status": "correct",
  "reasoning_label": "adequate_reasoning",
  "token_limit_hit_impact": "none"
}}

Allowed answer_status:
- "correct"
- "incorrect"
- "incomplete"
- "unclear"

Allowed reasoning_label:
- "over_reasoning"
- "under_reasoning"
- "adequate_reasoning"

Allowed token_limit_hit_impact:
- "none"
- "hit_but_correct"
- "hit_and_incomplete"
- "hit_and_wrong"

TOKEN LIMIT HIT:
{final_token_limit_hit}

ORIGINAL QUERY:
{query}

REFERENCE ANSWER:
{reference}

MODEL RESPONSE:
{response}

EXTRACTED REASONING TEXT:
{reasoning_text}

RAW RESPONSE:
{raw_response}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict math correctness and reasoning-efficiency judge. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=judge_max_tokens,
    )

    output = completion.choices[0].message.content.strip()
    parsed = extract_json_object(output)

    if not parsed:
        low = output.lower()
        eval_result = '"eval_result": true' in low or "eval_result: true" in low

        return {
            "eval_result": eval_result,
            "answer_status": "correct" if eval_result else "unclear",
            "reasoning_label": "adequate_reasoning",
            "token_limit_hit_impact": "none",
        }

    eval_result = bool(parsed.get("eval_result", False))
    answer_status = parsed.get("answer_status") or ("correct" if eval_result else "incorrect")
    reasoning_label = parsed.get("reasoning_label") or "adequate_reasoning"
    token_limit_hit_impact = parsed.get("token_limit_hit_impact") or "none"

    if reasoning_label not in {"over_reasoning", "under_reasoning", "adequate_reasoning"}:
        reasoning_label = "adequate_reasoning"

    return {
        "eval_result": eval_result,
        "answer_status": answer_status,
        "reasoning_label": reasoning_label,
        "token_limit_hit_impact": token_limit_hit_impact,
    }


def compute_summary(results):
    total = len(results)

    correct = sum(1 for x in results if x.get("eval_result") is True)
    incorrect = sum(1 for x in results if x.get("eval_result") is False)
    errors = sum(1 for x in results if x.get("judge_error"))

    token_hit = [x for x in results if x.get("final_token_limit_hit") is True]
    token_not_hit = [x for x in results if x.get("final_token_limit_hit") is False]

    reasoning_counts = Counter(x.get("reasoning_label", "missing") for x in results)
    answer_status_counts = Counter(x.get("answer_status", "missing") for x in results)
    token_impact_counts = Counter(x.get("token_limit_hit_impact", "missing") for x in results)

    by_reasoning_correctness = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0})
    for x in results:
        label = x.get("reasoning_label", "missing")
        by_reasoning_correctness[label]["total"] += 1
        if x.get("eval_result") is True:
            by_reasoning_correctness[label]["correct"] += 1
        elif x.get("eval_result") is False:
            by_reasoning_correctness[label]["incorrect"] += 1

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": correct / total if total else 0.0,

        "final_token_limit_hit": {
            "total_hit": len(token_hit),
            "hit_and_true": sum(1 for x in token_hit if x.get("eval_result") is True),
            "hit_and_false": sum(1 for x in token_hit if x.get("eval_result") is False),
            "hit_accuracy": (
                sum(1 for x in token_hit if x.get("eval_result") is True) / len(token_hit)
                if token_hit else 0.0
            ),
        },

        "final_token_limit_not_hit": {
            "total_not_hit": len(token_not_hit),
            "not_hit_and_true": sum(1 for x in token_not_hit if x.get("eval_result") is True),
            "not_hit_and_false": sum(1 for x in token_not_hit if x.get("eval_result") is False),
            "not_hit_accuracy": (
                sum(1 for x in token_not_hit if x.get("eval_result") is True) / len(token_not_hit)
                if token_not_hit else 0.0
            ),
        },

        "reasoning_efficiency": {
            "over_reasoning": reasoning_counts.get("over_reasoning", 0),
            "under_reasoning": reasoning_counts.get("under_reasoning", 0),
            "adequate_reasoning": reasoning_counts.get("adequate_reasoning", 0),
            "missing": reasoning_counts.get("missing", 0),
            "over_reasoning_rate": reasoning_counts.get("over_reasoning", 0) / total if total else 0.0,
            "under_reasoning_rate": reasoning_counts.get("under_reasoning", 0) / total if total else 0.0,
            "adequate_reasoning_rate": reasoning_counts.get("adequate_reasoning", 0) / total if total else 0.0,
        },

        "answer_status_counts": dict(answer_status_counts),
        "token_limit_hit_impact_counts": dict(token_impact_counts),
        "by_reasoning_correctness": dict(by_reasoning_correctness),
    }


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not found.")

    input_dir = os.path.dirname(args.input_file)
    input_stem = os.path.splitext(os.path.basename(args.input_file))[0]

    if args.output_file is None:
        args.output_file = os.path.join(
            input_dir,
            f"judge_result_{input_stem}.json"
        )

    summary_path = os.path.join(
        os.path.dirname(args.output_file),
        f"judge_result_{input_stem}_summary.json"
    )

    print("\nComprehensive Math + Reasoning Judge")
    print("Input:", args.input_file)
    print("Output:", args.output_file)
    print("Summary:", summary_path)
    print("Judge model:", args.judge_model)

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    data = load_json_or_jsonl(args.input_file)
    results = []

    if os.path.exists(args.output_file):
        try:
            results = load_json_or_jsonl(args.output_file)
            print(f"\nResuming from {len(results)}/{len(data)}")
        except Exception as e:
            print(f"Could not load existing output: {e}")
            results = []

    start_idx = len(results)

    try:
        for idx in tqdm(
            range(start_idx, len(data)),
            initial=start_idx,
            total=len(data),
        ):
            item = data[idx]

            query = get_query(item)
            reference = get_reference(item)
            response = get_response(item)
            raw_response = get_raw_response(item)
            reasoning_text = get_reasoning_text(item)

            final_token_limit_hit = detect_final_token_limit_hit(
                item,
                expected_final_max_tokens=args.expected_final_max_tokens,
            )

            try:
                judge = judge_answer_and_reasoning(
                    client=client,
                    model=args.judge_model,
                    query=query,
                    reference=reference,
                    response=response,
                    reasoning_text=reasoning_text,
                    raw_response=raw_response,
                    final_token_limit_hit=final_token_limit_hit,
                    judge_max_tokens=args.judge_max_tokens,
                )

                item["eval_result"] = judge["eval_result"]
                item["answer_status"] = judge["answer_status"]
                item["reasoning_label"] = judge["reasoning_label"]
                item["token_limit_hit_impact"] = judge["token_limit_hit_impact"]

                item["judge_model"] = args.judge_model
                item["final_token_limit_hit"] = final_token_limit_hit

            except Exception as e:
                error_msg = str(e)

                item["eval_result"] = False
                item["answer_status"] = "error"
                item["reasoning_label"] = "missing"
                item["token_limit_hit_impact"] = "none"
                item["judge_model"] = args.judge_model
                item["final_token_limit_hit"] = final_token_limit_hit

                if is_content_filter_error(error_msg):
                    item["judge_label"] = "content_filtered"
                else:
                    item["judge_label"] = "error"

                item["judge_error"] = error_msg

            results.append(item)

            save_json(args.output_file, results)
            save_json(summary_path, compute_summary(results))

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
        save_json(args.output_file, results)
        save_json(summary_path, compute_summary(results))
        return

    summary = compute_summary(results)

    save_json(summary_path, summary)

    print("\nFinished.")
    print(json.dumps(summary, indent=2))

    print(f"\nResults saved: {args.output_file}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()