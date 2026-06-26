import os
import re
import json
import time
import argparse
import sqlite3
from datetime import datetime
from typing import Annotated, List, TypedDict

from tqdm import tqdm
from datasets import load_dataset
from ddgs import DDGS

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver


MODEL_REGISTRY = {
    "qwen35-4b": {
        "model_id": "qwen35-4b",
        "model_name": "qwen35-4b",
        "hf_id": "Qwen/Qwen3.5-4B",
    },
    "llama31-8b": {
        "model_id": "llama31-8b",
        "model_name": "llama31-8b",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "deepseek-r1-7b": {
        "model_id": "deepseek-r1-7b",
        "model_name": "deepseek-r1-7b",
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_choice", type=str, default="qwen35-4b", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--tool_model_choice", type=str, default="qwen35-4b", choices=list(MODEL_REGISTRY.keys()))

    parser.add_argument("--tool_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--final_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")

    parser.add_argument("--dataset_id", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_name", type=str, default=None)

    parser.add_argument("--prompt_field", type=str, default=None)
    parser.add_argument("--answer_field", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--save_dir", type=str, default="agentic_ai_benchmark")

    return parser.parse_args()


args = parse_args()


def slugify_name(name: str) -> str:
    name = name.replace("/", "_").replace("-", "_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def to_float_or_none(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def avg(xs):
    clean = [to_float_or_none(x) for x in xs]
    clean = [x for x in clean if x is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


MODEL_ID = MODEL_REGISTRY[args.model_choice]["model_id"]
MODEL_NAME = MODEL_REGISTRY[args.model_choice]["model_name"]
HF_MODEL_ID = MODEL_REGISTRY[args.model_choice]["hf_id"]

TOOL_MODEL_ID = MODEL_REGISTRY[args.tool_model_choice]["model_id"]
TOOL_MODEL_NAME = MODEL_REGISTRY[args.tool_model_choice]["model_name"]
TOOL_HF_MODEL_ID = MODEL_REGISTRY[args.tool_model_choice]["hf_id"]

HF_DATASET_ID = args.dataset_id
HF_DATASET_CONFIG = args.dataset_config
HF_SPLIT = args.split

if args.dataset_name is None:
    DATASET_NAME = slugify_name(HF_DATASET_ID)
    if HF_DATASET_CONFIG:
        DATASET_NAME += f"_{slugify_name(HF_DATASET_CONFIG)}"
else:
    DATASET_NAME = slugify_name(args.dataset_name)

DEFENDER_NAME = "vanilla"

SAVE_DIR = os.path.join(
    args.save_dir,
    DEFENDER_NAME,
    f"tool_{TOOL_MODEL_NAME}",
    f"final_{MODEL_NAME}",
    DATASET_NAME,
)
os.makedirs(SAVE_DIR, exist_ok=True)

MEMORY_DB = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_memory.sqlite",
)

CHECKPOINT_DB = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_{DATASET_NAME}_checkpoints.sqlite",
)

TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVE_PATH = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_{DATASET_NAME}_{TIME_TAG}.json",
)

LATEST_PATH = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_{DATASET_NAME}_latest.json",
)

SUMMARY_PATH = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_{DATASET_NAME}_{TIME_TAG}_summary.json",
)

LATEST_SUMMARY_PATH = os.path.join(
    SAVE_DIR,
    f"{DEFENDER_NAME}_tool_{TOOL_MODEL_NAME}_final_{MODEL_NAME}_{DATASET_NAME}_latest_summary.json",
)


tool_llm = ChatOpenAI(
    model=TOOL_MODEL_ID,
    base_url=args.tool_base_url,
    api_key=args.api_key,
    temperature=0.0,
    max_tokens=args.max_tokens,
)

final_llm = ChatOpenAI(
    model=MODEL_ID,
    base_url=args.final_base_url,
    api_key=args.api_key,
    temperature=0.0,
    max_tokens=args.max_tokens,
)


def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(memory, content='memories', content_rowid='id')
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ai
        AFTER INSERT ON memories
        BEGIN
            INSERT INTO memories_fts(rowid, memory)
            VALUES (new.id, new.memory);
        END;
        """
    )

    conn.commit()
    conn.close()


init_memory_db()


def now_iso():
    return datetime.now().isoformat()


def make_tool_record(tool_name, tool_input, tool_output, start_time, end_time):
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "time_seconds": round(end_time - start_time, 4),
        "start_time": start_time,
        "end_time": end_time,
        "start_time_iso": datetime.fromtimestamp(start_time).isoformat(),
        "end_time_iso": datetime.fromtimestamp(end_time).isoformat(),
    }


@tool
def web_search(query: str) -> str:
    """Search the web when external or current information is needed."""
    start_time = time.time()

    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=2)

            for r in search_results:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", "")[:150],
                        "url": r.get("href", ""),
                    }
                )

        if not results:
            output = "No web search results found."
        else:
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"""
RESULT {i}

TITLE:
{r['title']}

SUMMARY:
{r['body']}

URL:
{r['url']}
"""
                )
            output = "\n\n".join(formatted)

    except Exception as e:
        output = f"Web search error: {str(e)}"

    end_time = time.time()

    return json.dumps(
        make_tool_record(
            "web_search",
            {"query": query},
            output,
            start_time,
            end_time,
        ),
        ensure_ascii=False,
    )


@tool
def save_memory(memory: str) -> str:
    """Save long-term memory when useful."""
    start_time = time.time()

    try:
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO memories (memory) VALUES (?)", (memory,))
        conn.commit()
        conn.close()
        output = f"Saved memory: {memory}"

    except Exception as e:
        output = f"Memory save error: {str(e)}"

    end_time = time.time()

    return json.dumps(
        make_tool_record(
            "save_memory",
            {"memory": memory},
            output,
            start_time,
            end_time,
        ),
        ensure_ascii=False,
    )


@tool
def search_memory(query: str) -> str:
    """Search long-term memory for previous user facts, preferences, or prior conversation information."""
    start_time = time.time()

    try:
        conn = sqlite3.connect(MEMORY_DB)
        cur = conn.cursor()

        clean_query = re.sub(r"[^a-zA-Z0-9\s]", " ", query).strip()
        words = [w for w in clean_query.split() if len(w) > 1]

        rows = []

        if words:
            fts_query = " OR ".join(words)

            cur.execute(
                """
                SELECT 
                    memories.memory,
                    memories.created_at,
                    bm25(memories_fts) AS score
                FROM memories_fts
                JOIN memories ON memories_fts.rowid = memories.id
                WHERE memories_fts MATCH ?
                ORDER BY score
                LIMIT 10
                """,
                (fts_query,),
            )

            rows = cur.fetchall()

        if not rows:
            like_conditions = []
            params = []

            for w in words:
                like_conditions.append("LOWER(memory) LIKE ?")
                params.append(f"%{w.lower()}%")

            if like_conditions:
                sql = f"""
                    SELECT memory, created_at, 999 AS score
                    FROM memories
                    WHERE {' OR '.join(like_conditions)}
                    ORDER BY created_at DESC
                    LIMIT 10
                """

                cur.execute(sql, params)
                rows = cur.fetchall()

        conn.close()

        if not rows:
            output = "No relevant memory found."
        else:
            output = "\n".join(
                [
                    f"- [{created_at}] {memory}"
                    for memory, created_at, _ in rows
                ]
            )

    except Exception as e:
        output = f"Memory search error: {str(e)}"

    end_time = time.time()

    return json.dumps(
        make_tool_record(
            "search_memory",
            {"query": query},
            output,
            start_time,
            end_time,
        ),
        ensure_ascii=False,
    )


tools = [web_search, save_memory, search_memory]
llm_with_tools = tool_llm.bind_tools(tools)
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def clean_response(text):
    if not text:
        return ""

    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<function=.*", "", text, flags=re.DOTALL)
    text = text.replace("Thinking Process:", "")

    return text.strip()


def extract_reasoning_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    if "</think>" in raw_text:
        return raw_text.split("</think>")[0].strip()

    return ""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def get_usage(msg):
    usage = getattr(msg, "usage_metadata", None)

    if usage:
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    response_metadata = getattr(msg, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) or {}

    return {
        "input_tokens": token_usage.get("prompt_tokens"),
        "output_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
    }


def extract_reasoning_tokens_from_metadata(msg):
    response_metadata = getattr(msg, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) or {}

    details = token_usage.get("completion_tokens_details", {}) or {}
    reasoning_tokens = details.get("reasoning_tokens")

    if reasoning_tokens is not None:
        return reasoning_tokens

    details = response_metadata.get("completion_tokens_details", {}) or {}
    return details.get("reasoning_tokens")


def build_token_report(raw_response, cleaned_response, response_msg):
    usage = get_usage(response_msg)

    visible_tokens_est = estimate_tokens(cleaned_response)
    reasoning_text = extract_reasoning_text(raw_response)
    reasoning_tokens_from_text_est = estimate_tokens(reasoning_text)

    reasoning_tokens_metadata = extract_reasoning_tokens_from_metadata(response_msg)
    output_tokens = usage.get("output_tokens")

    reasoning_tokens_est = None
    output_tokens_float = to_float_or_none(output_tokens)

    if output_tokens_float is not None:
        reasoning_tokens_est = max(output_tokens_float - visible_tokens_est, 0)

    hit_max_tokens = False
    if output_tokens_float is not None:
        hit_max_tokens = output_tokens_float >= args.max_tokens

    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "visible_tokens_est": visible_tokens_est,
        "reasoning_tokens_metadata": reasoning_tokens_metadata,
        "reasoning_tokens_from_text_est": reasoning_tokens_from_text_est,
        "reasoning_tokens_est": reasoning_tokens_est,
        "reasoning_text": reasoning_text,
        "reasoning_text_chars": len(reasoning_text),
        "visible_response_chars": len(cleaned_response or ""),
        "raw_response_chars": len(raw_response or ""),
        "hit_max_tokens": hit_max_tokens,
    }


def make_json_safe(obj):
    if isinstance(obj, str):
        return obj

    if isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, bool) or obj is None:
        return obj

    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, BaseMessage):
        item = {
            "type": obj.type,
            "content": obj.content,
        }

        if getattr(obj, "tool_calls", None):
            item["tool_calls"] = make_json_safe(obj.tool_calls)

        usage = get_usage(obj)
        if any(v is not None for v in usage.values()):
            item["usage"] = usage

        if getattr(obj, "additional_kwargs", None):
            item["additional_kwargs"] = make_json_safe(obj.additional_kwargs)

        return item

    return str(obj)


def extract_tool_records(messages):
    tool_records = []

    for m in messages:
        if getattr(m, "type", "") == "tool":
            content = str(m.content)

            try:
                parsed = json.loads(content)

                if isinstance(parsed, dict) and "tool_name" in parsed:
                    tool_records.append(parsed)
                else:
                    tool_records.append(
                        {
                            "tool_name": "unknown",
                            "tool_input": None,
                            "tool_output": content,
                            "time_seconds": None,
                        }
                    )

            except Exception:
                tool_records.append(
                    {
                        "tool_name": "unknown",
                        "tool_input": None,
                        "tool_output": content,
                        "time_seconds": None,
                    }
                )

    return tool_records


def extract_tool_calls(messages):
    tool_calls = []

    for m in messages:
        calls = getattr(m, "tool_calls", None)

        if calls:
            for c in calls:
                tool_calls.append(make_json_safe(c))

    return tool_calls


def build_main_chat_query(messages):
    user_query = ""
    tool_context = []

    for m in messages:
        if isinstance(m, HumanMessage):
            user_query = m.content

        if getattr(m, "type", "") == "tool":
            content = str(m.content)

            try:
                parsed = json.loads(content)

                if isinstance(parsed, dict) and "tool_output" in parsed:
                    tool_context.append(parsed["tool_output"])
                else:
                    tool_context.append(content)

            except Exception:
                tool_context.append(content)

    if tool_context:
        tool_text = "\n".join(tool_context)
        tool_text = tool_text[:3000]

        return f"""
Original user query:
{user_query}

Tool results:
{tool_text}

Use the tool results if they are relevant.

"""

    return user_query


def generate_final_response(query: str):
    final_query = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    messages = [
        (
            "system",
            "You are a helpful AI assistant.",
        ),
        (
            "human",
            query,
        ),
    ]

    final_start = time.time()

    response = final_llm.invoke(messages)

    final_end = time.time()

    raw_response = response.content
    final_response = clean_response(raw_response)

    token_report = build_token_report(
        raw_response=raw_response,
        cleaned_response=final_response,
        response_msg=response,
    )

    timing_report = {
        "final_inference_start_time": final_start,
        "final_inference_end_time": final_end,
        "final_inference_start_time_iso": datetime.fromtimestamp(final_start).isoformat(),
        "final_inference_end_time_iso": datetime.fromtimestamp(final_end).isoformat(),
        "final_inference_seconds": round(final_end - final_start, 4),
    }

    return final_response, final_query, raw_response, token_report, timing_report, response


def convert_messages_for_tool_model(messages):
    converted = []

    for m in messages:
        if isinstance(m, HumanMessage):
            converted.append(("human", m.content))
        elif isinstance(m, AIMessage):
            if m.content:
                converted.append(("ai", m.content))
        elif isinstance(m, SystemMessage):
            converted.append(("system", m.content))

    return converted


def tool_decision_node(state: AgentState):
    recent_messages = state["messages"][-10:]

    tuple_messages = [
        (
            "system",
            (
                "You are a tool router.\n"
                "Your job is ONLY to decide whether a tool is needed.\n"
                "Do not solve the user's problem.\n"
                "Do not provide explanations.\n"
                "Do not reason about the answer.\n\n"

                "Tool policy:\n"
                "- Never use tools for mathematics.\n"
                "- Never use tools for logical reasoning.\n"
                "- Never use tools for writing tasks.\n"
                "- Never use tools for general knowledge.\n"
                "- Use web_search only for current or external information.\n"
                "- Use search_memory only when prior memory is required.\n"
                "- Use save_memory only when explicitly asked.\n"
            )
        )
    ]

    tuple_messages.extend(convert_messages_for_tool_model(recent_messages))

    tool_decision_start = time.time()

    try:
        response = llm_with_tools.invoke(tuple_messages)

    except Exception as e:
        print(f"\nTool call error: {e}")

        response = AIMessage(
            content="",
            additional_kwargs={
                "tool_call_error": str(e),
            },
        )

    tool_decision_end = time.time()

    raw_tool_response = getattr(response, "content", "") or ""
    cleaned_tool_response = clean_response(raw_tool_response)

    tool_token_report = build_token_report(
        raw_response=raw_tool_response,
        cleaned_response=cleaned_tool_response,
        response_msg=response,
    )

    response.additional_kwargs["tool_decision_timing"] = {
        "tool_decision_start_time": tool_decision_start,
        "tool_decision_end_time": tool_decision_end,
        "tool_decision_start_time_iso": datetime.fromtimestamp(tool_decision_start).isoformat(),
        "tool_decision_end_time_iso": datetime.fromtimestamp(tool_decision_end).isoformat(),
        "tool_decision_seconds": round(tool_decision_end - tool_decision_start, 4),
    }

    response.additional_kwargs["tool_decision_raw_response"] = raw_tool_response
    response.additional_kwargs["tool_decision_clean_response"] = cleaned_tool_response
    response.additional_kwargs["tool_decision_token_report"] = tool_token_report
    response.additional_kwargs["tool_decision_usage"] = get_usage(response)
    response.additional_kwargs["tool_decision_has_tool_calls"] = bool(getattr(response, "tool_calls", None))

    return {"messages": [response]}


def main_chat_node(state: AgentState):
    query = build_main_chat_query(state["messages"])

    (
        final_response,
        final_query,
        raw_response,
        token_report,
        timing_report,
        model_response,
    ) = generate_final_response(query)

    response = AIMessage(
        content=final_response,
        additional_kwargs={
            "final_query": final_query,
            "raw_response": raw_response,
            "reasoning_text": token_report.get("reasoning_text", ""),
            "token_report": token_report,
            "usage": get_usage(model_response),
            "final_timing": timing_report,
        },
    )

    return {"messages": [response]}


def route_after_tool_decision(state: AgentState):
    last_msg = state["messages"][-1]

    if getattr(last_msg, "tool_calls", None):
        return "tools"

    return "main_chat"


checkpoint_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_conn)

graph = StateGraph(AgentState)

graph.add_node("tool_decision", tool_decision_node)
graph.add_node("tools", tool_node)
graph.add_node("main_chat", main_chat_node)

graph.set_entry_point("tool_decision")

graph.add_conditional_edges(
    "tool_decision",
    route_after_tool_decision,
    {
        "tools": "tools",
        "main_chat": "main_chat",
    },
)

graph.add_edge("tools", "main_chat")
graph.add_edge("main_chat", END)

app = graph.compile(checkpointer=checkpointer)


def extract_agent_timing(messages):
    timing = {
        "tool_decision_timing": None,
        "final_timing": None,
        "tool_execution_total_seconds": 0.0,
        "agent_total_recorded_seconds": 0.0,
    }

    for m in messages:
        additional = getattr(m, "additional_kwargs", None) or {}

        if "tool_decision_timing" in additional:
            timing["tool_decision_timing"] = additional["tool_decision_timing"]

        if "final_timing" in additional:
            timing["final_timing"] = additional["final_timing"]

    tool_records = extract_tool_records(messages)

    tool_total = 0.0
    for r in tool_records:
        value = to_float_or_none(r.get("time_seconds"))
        if value is not None:
            tool_total += value

    timing["tool_execution_total_seconds"] = round(tool_total, 4)

    total = 0.0

    if timing["tool_decision_timing"]:
        total += to_float_or_none(
            timing["tool_decision_timing"].get("tool_decision_seconds", 0.0)
        ) or 0.0

    if timing["final_timing"]:
        total += to_float_or_none(
            timing["final_timing"].get("final_inference_seconds", 0.0)
        ) or 0.0

    total += tool_total

    timing["agent_total_recorded_seconds"] = round(total, 4)

    return timing


def extract_tool_decision_report(messages):
    for m in messages:
        additional = getattr(m, "additional_kwargs", None) or {}

        if "tool_decision_timing" in additional:
            return {
                "tool_decision_raw_response": additional.get("tool_decision_raw_response", ""),
                "tool_decision_clean_response": additional.get("tool_decision_clean_response", ""),
                "tool_decision_has_tool_calls": additional.get("tool_decision_has_tool_calls", False),
                "tool_decision_usage": additional.get("tool_decision_usage", {}),
                "tool_decision_token_report": additional.get("tool_decision_token_report", {}),
                "tool_decision_timing": additional.get("tool_decision_timing", {}),
            }

    return {}


def save_results(results):
    tmp_latest = LATEST_PATH + ".tmp"
    tmp_save = SAVE_PATH + ".tmp"

    with open(tmp_latest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    os.replace(tmp_latest, LATEST_PATH)

    with open(tmp_save, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    os.replace(tmp_save, SAVE_PATH)


def save_summary(results):
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    tool_call_counts = [to_float_or_none(r.get("tool_call_count", 0)) or 0 for r in successful]
    tool_exec_counts = [to_float_or_none(r.get("tool_execution_count", 0)) or 0 for r in successful]
    latencies = [to_float_or_none(r.get("latency_seconds", 0)) or 0 for r in successful]

    final_times = []
    tool_decision_times = []
    tool_execution_times = []
    recorded_total_times = []

    reasoning_tokens = []
    output_tokens = []
    input_tokens = []

    for r in successful:
        timing = r.get("agent_timing_report", {}) or {}

        final_timing = timing.get("final_timing") or {}
        tool_decision_timing = timing.get("tool_decision_timing") or {}

        final_times.append(
            to_float_or_none(final_timing.get("final_inference_seconds", 0)) or 0
        )

        tool_decision_times.append(
            to_float_or_none(tool_decision_timing.get("tool_decision_seconds", 0)) or 0
        )

        tool_execution_times.append(
            to_float_or_none(timing.get("tool_execution_total_seconds", 0)) or 0
        )

        recorded_total_times.append(
            to_float_or_none(timing.get("agent_total_recorded_seconds", 0)) or 0
        )

        rt = to_float_or_none(r.get("reasoning_tokens_est"))
        ot = to_float_or_none(r.get("output_tokens"))
        it = to_float_or_none(r.get("input_tokens"))

        if rt is not None:
            reasoning_tokens.append(rt)

        if ot is not None:
            output_tokens.append(ot)

        if it is not None:
            input_tokens.append(it)

    summary = {
        "created_at": now_iso(),
        "dataset": DATASET_NAME,
        "hf_dataset_id": HF_DATASET_ID,
        "hf_dataset_config": HF_DATASET_CONFIG,
        "split": HF_SPLIT,
        "defense_name": DEFENDER_NAME,
        "tool_model_name": TOOL_MODEL_NAME,
        "tool_hf_model_id": TOOL_HF_MODEL_ID,
        "final_model_name": MODEL_NAME,
        "final_hf_model_id": HF_MODEL_ID,
        "total_saved": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_latency_seconds": avg(latencies),
        "avg_tool_decision_seconds": avg(tool_decision_times),
        "avg_tool_execution_seconds": avg(tool_execution_times),
        "avg_final_inference_seconds": avg(final_times),
        "avg_agent_total_recorded_seconds": avg(recorded_total_times),
        "avg_tool_call_count": avg(tool_call_counts),
        "avg_tool_execution_count": avg(tool_exec_counts),
        "avg_input_tokens": avg(input_tokens),
        "avg_output_tokens": avg(output_tokens),
        "avg_reasoning_tokens_est": avg(reasoning_tokens),
        "num_with_tool_calls": sum(1 for x in tool_call_counts if x > 0),
        "num_hit_max_tokens": sum(1 for r in successful if r.get("hit_max_tokens")),
        "save_path": SAVE_PATH,
        "latest_path": LATEST_PATH,
    }

    for path in [SUMMARY_PATH, LATEST_SUMMARY_PATH]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def load_existing_results():
    if os.path.exists(LATEST_PATH):
        try:
            with open(LATEST_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)

            if len(results) > 0 and "error" in results[-1]:
                print(
                    f"Removing failed index "
                    f"{results[-1].get('index')} "
                    f"and retrying it."
                )
                results = results[:-1]

            return results

        except Exception as e:
            print(f"Could not load previous results: {e}")
            return []

    return []


PROMPT_FIELDS = [
    "prompt",
    "query",
    "question",
    "Problem",
    "problem",
    "Pre-Revision Question",
    "Post-Revision Question",
    "goal",
    "goals",
    "instruction",
    "input",
    "task",
]

ANSWER_FIELDS = [
    "answer",
    "Answer",
    "solution",
    "Solution",
    "ground_truth",
    "ground_truth_plan",
    "Pre-Revision Correct Answer",
    "Post-Revision Correct Answer",
    "correct_answer",
    "target",
    "label",
]


def get_prompt(row):
    if args.prompt_field:
        if args.prompt_field not in row:
            raise KeyError(
                f"Prompt field '{args.prompt_field}' not found. "
                f"Available keys: {list(row.keys())}"
            )
        return str(row[args.prompt_field])

    if "final_query" in row:
        final_query = row["final_query"]

        if isinstance(final_query, list):
            for item in final_query:
                if isinstance(item, dict) and item.get("role") == "user":
                    return str(item.get("content", ""))
            return str(final_query)

        return str(final_query)

    for field in PROMPT_FIELDS:
        if field in row and row[field] is not None:
            return str(row[field])

    raise KeyError(f"No prompt field found. Available keys: {list(row.keys())}")


def get_reference_answer(row):
    if args.answer_field:
        return row.get(args.answer_field, None)

    for field in ANSWER_FIELDS:
        if field in row and row[field] is not None:
            return row[field]

    return None


def load_dataset_rows():
    if HF_DATASET_CONFIG:
        dataset = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, split=HF_SPLIT)
    else:
        dataset = load_dataset(HF_DATASET_ID, split=HF_SPLIT)

    rows = list(dataset)

    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    return rows


def make_dataset_thread_id(idx):
    return (
        f"{DATASET_NAME}-"
        f"{DEFENDER_NAME}-"
        f"tool_{TOOL_MODEL_NAME}-"
        f"final_{MODEL_NAME}-"
        f"{idx}"
    )


def run_dataset():
    data = load_dataset_rows()
    results = load_existing_results()

    completed_indices = set(
        item["index"]
        for item in results
        if "index" in item and "error" not in item
    )

    completed = len(completed_indices)

    print("=" * 80)
    print("Agentic AI Vanilla - Dataset Mode")
    print(f"Tool model via vLLM: {TOOL_MODEL_NAME}")
    print(f"Tool HF ID: {TOOL_HF_MODEL_ID}")
    print(f"Tool Base URL: {args.tool_base_url}")
    print(f"Final model via vLLM: {MODEL_NAME}")
    print(f"Final HF ID: {HF_MODEL_ID}")
    print(f"Final Base URL: {args.final_base_url}")
    print(f"Dataset: {HF_DATASET_ID}")
    print(f"Config: {HF_DATASET_CONFIG}")
    print(f"Split: {HF_SPLIT}")
    print(f"Dataset Name: {DATASET_NAME}")
    print(f"Defense: {DEFENDER_NAME}")
    print(f"Save Path: {SAVE_PATH}")
    print(f"Latest Path: {LATEST_PATH}")
    print(f"Summary Path: {SUMMARY_PATH}")
    print(f"Latest Summary Path: {LATEST_SUMMARY_PATH}")
    print(f"Memory DB: {MEMORY_DB}")
    print(f"Checkpoint DB: {CHECKPOINT_DB}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Total Samples: {len(data)}")
    print(f"Already Completed Successful Samples: {completed}")
    print("=" * 80)

    run_start = time.time()

    for idx in tqdm(range(len(data))):
        if idx in completed_indices:
            continue

        row = data[idx]
        original_query = get_prompt(row)
        reference_answer = get_reference_answer(row)

        dataset_thread_id = make_dataset_thread_id(idx)

        config = {
            "configurable": {
                "thread_id": dataset_thread_id
            }
        }

        sample_start = time.time()

        try:
            result = app.invoke(
                {
                    "messages": [
                        HumanMessage(content=original_query)
                    ]
                },
                config=config,
            )

            sample_end = time.time()
            latency = sample_end - sample_start

            final_msg = result["messages"][-1]
            final_response = clean_response(getattr(final_msg, "content", ""))

            tool_calls = extract_tool_calls(result["messages"])
            tool_records = extract_tool_records(result["messages"])
            tool_decision_report = extract_tool_decision_report(result["messages"])
            agent_timing_report = extract_agent_timing(result["messages"])

            token_report = final_msg.additional_kwargs.get("token_report", {})
            final_query = final_msg.additional_kwargs.get("final_query", original_query)

            output = dict(row)
            output["index"] = idx
            output["thread_id"] = dataset_thread_id

            output["tool_model_name"] = TOOL_MODEL_NAME
            output["tool_hf_model_id"] = TOOL_HF_MODEL_ID

            output["final_model_name"] = MODEL_NAME
            output["final_hf_model_id"] = HF_MODEL_ID

            output["model_name"] = MODEL_NAME
            output["hf_model_id"] = HF_MODEL_ID

            output["defense_name"] = DEFENDER_NAME
            output["dataset"] = DATASET_NAME
            output["hf_dataset_id"] = HF_DATASET_ID
            output["hf_dataset_config"] = HF_DATASET_CONFIG
            output["split"] = HF_SPLIT

            output["original_query"] = original_query
            output["reference_answer"] = make_json_safe(reference_answer)

            output["final_query"] = make_json_safe(final_query)
            output["final_response"] = final_response
            output["raw_response"] = final_msg.additional_kwargs.get("raw_response", "")
            output["reasoning_text"] = final_msg.additional_kwargs.get("reasoning_text", "")
            output["token_report"] = make_json_safe(token_report)

            output["input_tokens"] = token_report.get("input_tokens")
            output["output_tokens"] = token_report.get("output_tokens")
            output["total_tokens"] = token_report.get("total_tokens")
            output["visible_tokens_est"] = token_report.get("visible_tokens_est")
            output["reasoning_tokens_metadata"] = token_report.get("reasoning_tokens_metadata")
            output["reasoning_tokens_from_text_est"] = token_report.get("reasoning_tokens_from_text_est")
            output["reasoning_tokens_est"] = token_report.get("reasoning_tokens_est")
            output["reasoning_text_chars"] = token_report.get("reasoning_text_chars")
            output["visible_response_chars"] = token_report.get("visible_response_chars")
            output["raw_response_chars"] = token_report.get("raw_response_chars")
            output["hit_max_tokens"] = token_report.get("hit_max_tokens")

            output["tool_call_count"] = len(tool_calls)
            output["tool_calls"] = make_json_safe(tool_calls)

            output["tool_execution_count"] = len(tool_records)
            output["tool_records"] = make_json_safe(tool_records)

            output["tool_decision_report"] = make_json_safe(tool_decision_report)
            output["agent_timing_report"] = make_json_safe(agent_timing_report)

            output["sample_start_time"] = sample_start
            output["sample_end_time"] = sample_end
            output["sample_start_time_iso"] = datetime.fromtimestamp(sample_start).isoformat()
            output["sample_end_time_iso"] = datetime.fromtimestamp(sample_end).isoformat()
            output["latency_seconds"] = round(latency, 4)

            output["message_trace"] = make_json_safe(result["messages"])

            results = [item for item in results if item.get("index") != idx]
            results.append(output)
            results = sorted(results, key=lambda x: x.get("index", -1))

            save_results(results)
            save_summary(results)

            completed_indices.add(idx)

        except Exception as e:
            sample_end = time.time()
            latency = sample_end - sample_start

            error_output = dict(row)
            error_output["index"] = idx
            error_output["thread_id"] = dataset_thread_id

            error_output["tool_model_name"] = TOOL_MODEL_NAME
            error_output["tool_hf_model_id"] = TOOL_HF_MODEL_ID

            error_output["final_model_name"] = MODEL_NAME
            error_output["final_hf_model_id"] = HF_MODEL_ID

            error_output["model_name"] = MODEL_NAME
            error_output["hf_model_id"] = HF_MODEL_ID

            error_output["defense_name"] = DEFENDER_NAME
            error_output["dataset"] = DATASET_NAME
            error_output["hf_dataset_id"] = HF_DATASET_ID
            error_output["hf_dataset_config"] = HF_DATASET_CONFIG
            error_output["split"] = HF_SPLIT

            error_output["original_query"] = original_query
            error_output["reference_answer"] = make_json_safe(reference_answer)

            error_output["sample_start_time"] = sample_start
            error_output["sample_end_time"] = sample_end
            error_output["sample_start_time_iso"] = datetime.fromtimestamp(sample_start).isoformat()
            error_output["sample_end_time_iso"] = datetime.fromtimestamp(sample_end).isoformat()
            error_output["latency_seconds"] = round(latency, 4)

            error_output["error"] = str(e)

            results = [item for item in results if item.get("index") != idx]
            results.append(error_output)
            results = sorted(results, key=lambda x: x.get("index", -1))

            save_results(results)
            save_summary(results)

            print(f"\nError at index {idx}: {e}")
            print("Continuing to next sample.")

            continue

    run_end = time.time()

    print("\nFinished.")
    print(f"Run seconds: {round(run_end - run_start, 4)}")
    print(f"Saved latest: {LATEST_PATH}")
    print(f"Saved timestamped: {SAVE_PATH}")
    print(f"Saved latest summary: {LATEST_SUMMARY_PATH}")
    print(f"Saved timestamped summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_dataset()
