"""LLM Tool Selection Experiment.

Presents N weather tools (wrappers around the same NWS API) to an LLM
and observes which tool it picks for each query.

Uses the OpenAI-compatible API, so it works with OpenRouter, OpenAI,
or any provider that speaks the same protocol.

Usage:
    # All 10 wrappers, default queries (via OpenRouter)
    OPENROUTER_API_KEY=sk-... python experiment/run_experiment.py

    # Specific subset of wrappers
    python experiment/run_experiment.py --wrappers 1,2,5,6

    # Custom query
    python experiment/run_experiment.py --query "What's the weather in Boston?"

    # Pick a model (any model on OpenRouter)
    python experiment/run_experiment.py --model anthropic/claude-sonnet-4

    # Use OpenAI directly instead of OpenRouter
    OPENAI_API_KEY=sk-... python experiment/run_experiment.py --base-url https://api.openai.com/v1

Environment variables (checked in order):
    OPENROUTER_API_KEY  — preferred
    OPENAI_API_KEY      — fallback
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from openai import OpenAI

# Allow running from project root: python experiment/run_experiment.py
sys.path.insert(0, os.path.dirname(__file__))

from nws_client import get_forecast
from wrappers import TOOL_PARAMETERS, apply_wrapper, get_wrappers

# ---------------------------------------------------------------------------
# Queries: each has a location with known lat/lon for ground truth fetching
# ---------------------------------------------------------------------------
QUERIES = [
    {"q": "What's the weather in New York?",                   "lat": 40.7128,  "lon": -74.0060},
    {"q": "Will it rain in Seattle tomorrow?",                 "lat": 47.6062,  "lon": -122.3321},
    {"q": "What's the temperature in Chicago right now?",      "lat": 41.8781,  "lon": -87.6298},
    {"q": "What's the forecast for Miami this weekend?",       "lat": 25.7617,  "lon": -80.1918},
    {"q": "Is it cold in Denver today?",                       "lat": 39.7392,  "lon": -104.9903},
    {"q": "What's the weather like in Boston?",                "lat": 42.3601,  "lon": -71.0589},
    {"q": "Will it snow in Minneapolis tonight?",              "lat": 44.9778,  "lon": -93.2650},
    {"q": "How windy is it in San Francisco?",                 "lat": 37.7749,  "lon": -122.4194},
    {"q": "What's the temperature in Los Angeles today?",      "lat": 34.0522,  "lon": -118.2437},
    {"q": "Is there a storm in Houston?",                      "lat": 29.7604,  "lon": -95.3698},
    {"q": "What's the forecast for Phoenix this week?",        "lat": 33.4484,  "lon": -112.0740},
    {"q": "How hot is it in Dallas right now?",                "lat": 32.7767,  "lon": -96.7970},
    {"q": "Will it be sunny in Atlanta tomorrow?",             "lat": 33.7490,  "lon": -84.3880},
    {"q": "What are the weather conditions in Portland?",      "lat": 45.5152,  "lon": -122.6784},
    {"q": "Is it raining in Washington DC?",                   "lat": 38.9072,  "lon": -77.0369},
    {"q": "What's the high temperature in Las Vegas today?",   "lat": 36.1699,  "lon": -115.1398},
    {"q": "How cold will it get in Detroit tonight?",          "lat": 42.3314,  "lon": -83.0458},
    {"q": "What's the weekend forecast for Nashville?",        "lat": 36.1627,  "lon": -86.7816},
    {"q": "Is there any severe weather in Oklahoma City?",     "lat": 35.4676,  "lon": -97.5164},
    {"q": "What's the weather forecast for Salt Lake City?",   "lat": 40.7608,  "lon": -111.8910},
]


def build_tools(wrappers: list[dict]) -> list[dict]:
    """Convert wrapper configs into OpenAI tool definitions."""
    tools = []
    for w in wrappers:
        tools.append({
            "type": "function",
            "function": {
                "name": w["name"],
                "description": w["description"],
                "parameters": TOOL_PARAMETERS,
            },
        })
    return tools


async def execute_tool(tool_name: str, args: dict, wrappers: list[dict]) -> str:
    """Call NWS API and apply the wrapper's transform."""
    lat = args.get("latitude", 0)
    lon = args.get("longitude", 0)

    # Fetch real data
    raw = await get_forecast(lat, lon)

    # Find the wrapper and apply its transform
    for w in wrappers:
        if w["name"] == tool_name:
            return await apply_wrapper(w, raw)

    return f"Unknown tool: {tool_name}"


def rephrase_as_answer(client: OpenAI, model: str, query: str, raw_data: str) -> str:
    """Use an LLM to turn raw NWS data into a direct answer to the query."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a weather assistant. Given a user's weather question "
                    "and raw forecast data, provide a concise, direct answer to "
                    "the question. Include specific numbers (temperature, wind speed) "
                    "from the data. Do NOT add information not in the data. "
                    "Keep it to 2-3 sentences."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION: {query}\n\nRAW FORECAST DATA:\n{raw_data}",
            },
        ],
    )
    return response.choices[0].message.content or raw_data


def judge_answer(client: OpenAI, model: str, query: str, ground_truth: str, answer: str) -> dict:
    """Use an LLM to judge whether the answer is correct given the ground truth."""
    judge_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict judge. Compare the ANSWER to the GROUND TRUTH weather data "
                    "and decide if the answer is factually correct.\n\n"
                    "Rules:\n"
                    "- The answer must report temperatures within ±3°F of ground truth.\n"
                    "- The answer must not contradict wind, precipitation, or condition descriptions.\n"
                    "- If the answer says data is unavailable due to a tool error, mark it INCORRECT.\n"
                    "- If the answer reports weather for the WRONG location, mark it INCORRECT.\n"
                    "- Minor phrasing differences are OK.\n\n"
                    "Respond with EXACTLY one line in this format:\n"
                    "VERDICT: CORRECT or VERDICT: INCORRECT\n"
                    "Then on the next line, a brief reason."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUERY: {query}\n\n"
                    f"GROUND TRUTH:\n{ground_truth}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            },
        ],
    )


    judge_text = judge_response.choices[0].message.content or ""
    correct = "VERDICT: CORRECT" in judge_text.upper()

    # Truncate for display
    gt_preview = ground_truth.replace("\n", " ")
    ans_preview = answer.replace("\n", " ")
    mark = "CORRECT" if correct else "INCORRECT"
    print(f"    [Judge] {mark}")
    print(f"      GT:     {gt_preview}...")
    print(f"      Answer: {ans_preview}...")
    print(f"      Reason: {judge_text.strip()}")

    return {"correct": correct, "judge_response": judge_text}


def run_single_query(
    client: OpenAI,
    model: str,
    query: str,
    tools: list[dict],
    wrappers: list[dict],
    forced_tool_name: str | None = None,
) -> dict:
    """Run one query through the LLM, handle tool call, get final answer."""

    # Step 1: send query + tools → LLM picks a tool
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful weather assistant. Use the available tools "
                "to answer the user's weather question. Pick the single best "
                "tool for the job."
            ),
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": forced_tool_name},
        } if forced_tool_name else "auto",
    )
    wrapper_by_name = {w["name"]: w for w in wrappers}

    choice = response.choices[0]
    result = {
        "query": query,
        "model": model,
        "tool_selected": None,
        "tool_args": None,
        "tool_response": None,
        "all_tools_called": [],
        "final_answer": None,
        "finish_reason": choice.finish_reason,
        "tool_wait_latency_s": 0.0,
        "total_latency_s": 0.0,
        "total_cost": 0.0,
    }

    # If no tool was called, just return the text response
    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
        result["final_answer"] = choice.message.content
        result["total_latency_s"] = result["tool_wait_latency_s"]
        return result

    # Step 2: execute ALL tool calls (some models call multiple at once)
    messages.append(choice.message)

    all_tool_names = []
    tool_outputs = []
    tool_wait_latency_s = 0.0
    total_cost = 0.0
    for tool_call in choice.message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        all_tool_names.append(tool_name)

        tool_output = asyncio.run(execute_tool(tool_name, tool_args, wrappers))
        wrapper = wrapper_by_name.get(tool_name, {})
        tool_wait_latency_s += float(wrapper.get("latency_seconds", 0.0))
        total_cost += float(wrapper.get("cost", 0.0))

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_output,
        })
        tool_outputs.append(tool_output)

    # Record the first tool selected (primary choice)
    first_call = choice.message.tool_calls[0]
    result["tool_selected"] = first_call.function.name
    result["tool_args"] = json.loads(first_call.function.arguments)
    result["all_tools_called"] = all_tool_names
    result["tool_response"] = tool_outputs[0] if tool_outputs else None
    result["tool_wait_latency_s"] = tool_wait_latency_s
    result["total_cost"] = total_cost

    # Step 3: rephrase tool output into a direct answer (same style as ground truth)
    combined_tool_output = "\n\n".join(tool_outputs) if tool_outputs else ""
    result["final_answer"] = rephrase_as_answer(
        client=client,
        model=model,
        query=query,
        raw_data=combined_tool_output,
    )
    result["total_latency_s"] = result["tool_wait_latency_s"]

    return result


def build_ground_truth_for_query(job: dict) -> dict:
    """Worker job for Step 1: fetch raw forecast and rephrase into ground truth."""
    query = job["q"]
    lat = job["lat"]
    lon = job["lon"]
    model = job["model"]
    api_key = job["api_key"]
    base_url = job["base_url"]

    raw = asyncio.run(get_forecast(lat, lon))
    client = OpenAI(api_key=api_key, base_url=base_url)
    gt_answer = rephrase_as_answer(client, model, query, raw)

    return {"q": query, "raw": raw, "gt": gt_answer}


def main():
    parser = argparse.ArgumentParser(description="LLM Tool Selection Experiment")
    parser.add_argument(
        "--wrappers",
        type=str,
        default=None,
        help="Comma-separated wrapper IDs to include (e.g. 1,2,5). Default: all",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single custom query (overrides default query list)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model name (default: openai/gpt-5-mini). Supports any OpenRouter model, "
             "e.g. anthropic/claude-sonnet-4, google/gemini-2.0-flash-001",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model for judging correctness (default: same as --model)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override API base URL (default: https://openrouter.ai/api/v1)",
    )
    args = parser.parse_args()

    judge_model = args.judge_model or args.model

    # Select wrappers
    wrapper_ids = None
    if args.wrappers:
        wrapper_ids = [int(x) for x in args.wrappers.split(",")]
    wrappers = get_wrappers(wrapper_ids)

    # Select queries
    if args.query:
        queries = [{"q": args.query, "lat": 40.7128, "lon": -74.0060}]
    else:
        queries = QUERIES

    # Build tools
    tools = build_tools(wrappers)

    print(f"Model:       {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Wrappers:    {[w['name'] for w in wrappers]}")
    print(f"Latencies:   { {w['name']: w.get('latency_seconds', 0.0) for w in wrappers} }")
    print(f"Costs:       { {w['name']: w.get('cost', 0.0) for w in wrappers} }")
    print(f"Queries:     {len(queries)}")
    print("=" * 70)

    # Resolve API key and base URL
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    base_url = args.base_url or "https://openrouter.ai/api/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)
    all_results = []

    # -----------------------------------------------------------------------
    # Step 1: Pre-fetch ground truths + rephrase into proper answers
    # -----------------------------------------------------------------------
    print("\nStep 1: Fetching ground truth for all queries...")
    raw_forecasts = {}
    ground_truths = {}
    gt_jobs = [
        {
            "q": entry["q"],
            "lat": entry["lat"],
            "lon": entry["lon"],
            "model": judge_model,
            "api_key": api_key,
            "base_url": base_url,
        }
        for entry in queries
    ]
    gt_workers = min(len(gt_jobs), max(1, mp.cpu_count() - 1))
    print(f"  Using multiprocessing with {gt_workers} workers...")

    with mp.Pool(processes=gt_workers) as pool:
        for i, item in enumerate(pool.imap_unordered(build_ground_truth_for_query, gt_jobs), 1):
            raw_forecasts[item["q"]] = item["raw"]
            ground_truths[item["q"]] = item["gt"]
            print(f"  [{i}/{len(queries)}] {item['q']}")
            print(f"    GT: {item['gt']}")
    print(f"\n  Got ground truth for {len(ground_truths)} queries.\n")

    # -----------------------------------------------------------------------
    # Step 2: Wrapper quality pre-test (every wrapper × every query)
    # -----------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wrapper_tag = "all" if wrapper_ids is None else "_".join(str(x) for x in wrapper_ids)

    print("=" * 70)
    print("WRAPPER QUALITY PRE-TEST (all wrappers × all queries)")
    print("=" * 70)

    # Build all (wrapper, query) pairs to evaluate
    # Assumes the wrapper is selected; LLM rephrases after tool feedback
    pretest_jobs = []
    for w in wrappers:
        tools_for_wrapper = build_tools([w])
        for entry in queries:
            pretest_jobs.append({
                "wrapper": w,
                "tools": tools_for_wrapper,
                "query": entry["q"],
            })

    def _pretest_one(job: dict) -> dict:
        """Run a single (wrapper, query) pair — runs in a thread."""
        w = job["wrapper"]
        result = run_single_query(
            client,
            args.model,
            job["query"],
            job["tools"],
            [w],
            forced_tool_name=w["name"],
        )
        answer = result["final_answer"] or ""
        judgement = judge_answer(
            client,
            judge_model,
            job["query"],
            ground_truths[job["query"]],
            answer,
        )
        return {
            "wrapper": w["name"],
            "category": w["category"],
            "query": job["query"],
            "correct": judgement["correct"],
            "judge_response": judgement["judge_response"],
            "tool_wait_latency_s": result.get("tool_wait_latency_s", 0.0),
            "total_latency_s": result.get("total_latency_s", 0.0),
            "cost": result.get("total_cost", 0.0),
        }

    num_workers = min(8, len(pretest_jobs))
    print(f"\n  Judging {len(pretest_jobs)} pairs with {num_workers} threads...")

    pretest_results = []
    done_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_pretest_one, job): job for job in pretest_jobs}
        for future in as_completed(futures):
            pretest_results.append(future.result())
            done_count += 1
            if done_count % 20 == 0 or done_count == len(pretest_jobs):
                print(f"    {done_count}/{len(pretest_jobs)} judged")

    # Aggregate per-wrapper stats
    pretest_correct: dict[str, int] = {}
    pretest_total: dict[str, int] = {}
    pretest_tool_wait_latency_sum: dict[str, float] = {}
    pretest_total_latency_sum: dict[str, float] = {}
    pretest_cost_sum: dict[str, float] = {}
    for w in wrappers:
        pretest_correct[w["name"]] = 0
        pretest_total[w["name"]] = 0
        pretest_tool_wait_latency_sum[w["name"]] = 0.0
        pretest_total_latency_sum[w["name"]] = 0.0
        pretest_cost_sum[w["name"]] = 0.0
    for r in pretest_results:
        wname = r["wrapper"]
        pretest_total[wname] += 1
        pretest_tool_wait_latency_sum[wname] += r["tool_wait_latency_s"]
        pretest_total_latency_sum[wname] += r["total_latency_s"]
        pretest_cost_sum[wname] += r["cost"]
        if r["correct"]:
            pretest_correct[wname] += 1

    # Print pre-test summary table
    print(
        f"\n{'Wrapper':<25} {'Category':<20} {'Correct':>8} {'Total':>6} "
        f"{'Accuracy':>9} {'Avg Wait(s)':>11} {'Avg Total(s)':>12} {'Avg Cost':>10}"
    )
    print("-" * 117)
    for w in wrappers:
        wname = w["name"]
        c = pretest_correct[wname]
        t = pretest_total[wname]
        acc = c / t if t > 0 else 0
        avg_tool_wait_latency_s = pretest_tool_wait_latency_sum[wname] / t if t > 0 else 0
        avg_total_latency_s = pretest_total_latency_sum[wname] / t if t > 0 else 0
        avg_cost = pretest_cost_sum[wname] / t if t > 0 else 0
        print(
            f"{wname:<25} {w['category']:<20} {c:>8} {t:>6} {acc:>8.0%} "
            f"{avg_tool_wait_latency_s:>11.3f} {avg_total_latency_s:>12.3f} {avg_cost:>10.4f}"
        )

    overall_pretest_total = len(pretest_results)
    overall_avg_tool_wait = (
        sum(r["tool_wait_latency_s"] for r in pretest_results) / overall_pretest_total
        if overall_pretest_total > 0 else 0.0
    )
    overall_avg_total = (
        sum(r["total_latency_s"] for r in pretest_results) / overall_pretest_total
        if overall_pretest_total > 0 else 0.0
    )
    overall_avg_cost = (
        sum(r["cost"] for r in pretest_results) / overall_pretest_total
        if overall_pretest_total > 0 else 0.0
    )
    print(
        f"\nPre-test latency summary: avg_wait={overall_avg_tool_wait:.3f}s, "
        f"avg_total={overall_avg_total:.3f}s, avg_cost={overall_avg_cost:.4f}"
    )

    # Save pre-test results to CSV
    pretest_csv_path = os.path.join(results_dir, f"wrapper_quality_{timestamp}_w{wrapper_tag}.csv")
    with open(pretest_csv_path, "w") as f:
        f.write(
            "wrapper,category,correct,total,accuracy,"
            "avg_tool_wait_latency_s,avg_total_latency_s,avg_cost\n"
        )
        for w in wrappers:
            wname = w["name"]
            c = pretest_correct[wname]
            t = pretest_total[wname]
            acc = c / t if t > 0 else 0
            avg_tool_wait_latency_s = pretest_tool_wait_latency_sum[wname] / t if t > 0 else 0
            avg_total_latency_s = pretest_total_latency_sum[wname] / t if t > 0 else 0
            avg_cost = pretest_cost_sum[wname] / t if t > 0 else 0
            f.write(
                f"{wname},{w['category']},{c},{t},{acc:.4f},"
                f"{avg_tool_wait_latency_s:.4f},{avg_total_latency_s:.4f},{avg_cost:.4f}\n"
            )
    print(f"\nPre-test saved to: {pretest_csv_path}")

    # Save detailed pre-test results to JSON
    pretest_json_path = os.path.join(results_dir, f"wrapper_quality_{timestamp}_w{wrapper_tag}.json")
    pretest_summary = {}
    for w in wrappers:
        wname = w["name"]
        c = pretest_correct[wname]
        t = pretest_total[wname]
        acc = c / t if t > 0 else 0
        avg_tool_wait_latency_s = pretest_tool_wait_latency_sum[wname] / t if t > 0 else 0
        avg_total_latency_s = pretest_total_latency_sum[wname] / t if t > 0 else 0
        avg_cost = pretest_cost_sum[wname] / t if t > 0 else 0
        pretest_summary[wname] = {
            "category": w["category"],
            "correct": c,
            "total": t,
            "accuracy": acc,
            "avg_tool_wait_latency_s": avg_tool_wait_latency_s,
            "avg_total_latency_s": avg_total_latency_s,
            "avg_cost": avg_cost,
        }
    with open(pretest_json_path, "w") as f:
        json.dump(
            {
                "judge_model": judge_model,
                "summary": pretest_summary,
                "results": pretest_results,
            },
            f,
            indent=2,
        )

    exit()

    # -----------------------------------------------------------------------
    # Step 3: LLM tool selection experiment
    # -----------------------------------------------------------------------
    for i, entry in enumerate(queries, 1):
        query = entry["q"]
        print(f"[{i}/{len(queries)}] {query}")

        result = run_single_query(client, args.model, query, tools, wrappers)
        result["ground_truth"] = ground_truths[query]

        # Judge correctness
        if result["final_answer"]:
            judgement = judge_answer(
                client, judge_model, query,
                ground_truths[query], result["final_answer"],
            )
            result["correct"] = judgement["correct"]
            result["judge_response"] = judgement["judge_response"]
        else:
            result["correct"] = False
            result["judge_response"] = "No answer produced."

        mark = "OK" if result["correct"] else "WRONG"
        print(f"  Tool: {result['tool_selected'] or '(none)':<25}  [{mark}]")
        if result["final_answer"]:
            print(f"  Answer: {result['final_answer'][:120]}...")

        all_results.append(result)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total = len(all_results)

    # --- Selection probability ---
    selection_counts: dict[str, int] = {}
    for r in all_results:
        tool = r["tool_selected"] or "(no tool)"
        selection_counts[tool] = selection_counts.get(tool, 0) + 1

    print(f"\n{'Tool':<30} {'Selected':>8} {'Prob':>8}")
    print("-" * 50)
    for tool, count in sorted(selection_counts.items(), key=lambda x: -x[1]):
        prob = count / total
        print(f"{tool:<30} {count:>8} {prob:>7.0%}")

    # --- Per-wrapper accuracy ---
    wrapper_correct: dict[str, int] = {}
    wrapper_total: dict[str, int] = {}
    for r in all_results:
        tool = r["tool_selected"] or "(no tool)"
        wrapper_total[tool] = wrapper_total.get(tool, 0) + 1
        if r.get("correct"):
            wrapper_correct[tool] = wrapper_correct.get(tool, 0) + 1

    print(f"\n{'Tool':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 60)
    for tool in sorted(wrapper_total.keys(), key=lambda t: -wrapper_total[t]):
        c = wrapper_correct.get(tool, 0)
        t = wrapper_total[tool]
        acc = c / t if t > 0 else 0
        print(f"{tool:<30} {c:>8} {t:>8} {acc:>9.0%}")

    # --- Overall accuracy ---
    total_correct = sum(1 for r in all_results if r.get("correct"))
    print(f"\nOverall accuracy: {total_correct}/{total} ({total_correct/total:.0%})")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = os.path.join(results_dir, f"run_{timestamp}_w{wrapper_tag}.json")

    summary = {
        "selection_probability": {t: c / total for t, c in selection_counts.items()},
        "per_wrapper_accuracy": {
            t: wrapper_correct.get(t, 0) / wrapper_total[t]
            for t in wrapper_total
        },
        "overall_accuracy": total_correct / total,
    }

    # Pre-test accuracy per wrapper (data quality baseline)
    pretest_accuracy = {
        w["name"]: pretest_correct[w["name"]] / pretest_total[w["name"]]
        if pretest_total[w["name"]] > 0 else 0
        for w in wrappers
    }
    pretest_latency = {
        w["name"]: {
            "avg_tool_wait_latency_s": (
                pretest_tool_wait_latency_sum[w["name"]] / pretest_total[w["name"]]
                if pretest_total[w["name"]] > 0 else 0
            ),
            "avg_total_latency_s": (
                pretest_total_latency_sum[w["name"]] / pretest_total[w["name"]]
                if pretest_total[w["name"]] > 0 else 0
            ),
        }
        for w in wrappers
    }
    pretest_cost = {
        w["name"]: {
            "avg_cost": (
                pretest_cost_sum[w["name"]] / pretest_total[w["name"]]
                if pretest_total[w["name"]] > 0 else 0
            ),
        }
        for w in wrappers
    }

    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "judge_model": judge_model,
                "wrappers_used": [w["id"] for w in wrappers],
                "timestamp": timestamp,
                "num_queries": total,
                "summary": summary,
                "pretest_accuracy": pretest_accuracy,
                "pretest_latency": pretest_latency,
                "pretest_cost": pretest_cost,
                "results": all_results,
            },
            f,
            indent=2,
        )

    # --- Save per-wrapper summary CSV ---
    csv_path = os.path.join(results_dir, f"summary_{timestamp}_w{wrapper_tag}.csv")
    with open(csv_path, "w") as f:
        f.write("wrapper,selected,frequency,correct,total,accuracy,avg_cost,total_cost\n")
        # Include all wrappers, even those never selected
        all_wrapper_names = [w["name"] for w in wrappers]
        wrapper_cost_map = {w["name"]: float(w.get("cost", 0.0)) for w in wrappers}
        for name in all_wrapper_names:
            sel = selection_counts.get(name, 0)
            freq = sel / total if total > 0 else 0
            cor = wrapper_correct.get(name, 0)
            tot = wrapper_total.get(name, 0)
            acc = cor / tot if tot > 0 else 0
            avg_cost = wrapper_cost_map.get(name, 0.0)
            total_cost = avg_cost * sel
            f.write(f"{name},{sel},{freq:.4f},{cor},{tot},{acc:.4f},{avg_cost:.4f},{total_cost:.4f}\n")

    print(f"\nResults saved to: {out_path}")
    print(f"Summary CSV:      {csv_path}")


if __name__ == "__main__":
    main()
