"""LLM request helpers for structured abstract evaluation."""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from pydantic import validate_call


load_dotenv()


def json_from_response(llm_output):
    """Extract JSON from a model response."""
    if isinstance(llm_output, (dict, list)):
        return llm_output

    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    cleaned = str(llm_output).strip()
    fence_match = re.search(
        r"^\s*`{1,3}\s*(?:json)?\s*(.*?)`{1,3}\s*$",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_content = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

    try:
        corrupted_dir = Path.cwd() / "corrupted_llm_outputs"
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        output_file = corrupted_dir / f"corrupted_llm_output_{int(time.time())}.txt"
        output_file.write_text(str(llm_output), encoding="utf-8")
    except Exception:
        pass

    return cleaned


def _parse_model_output(parser, content: str):
    """Parse a model response with a JSON fallback."""
    try:
        return parser.parse(content)
    except Exception:
        parsed_json = json_from_response(content)
        if hasattr(parser, "parse_obj"):
            return parser.parse_obj(parsed_json)
        return parser.pydantic_object.model_validate(parsed_json)


def _model_to_dict(model_output) -> dict:
    """Convert parser output to a plain dictionary."""
    if isinstance(model_output, dict):
        return model_output
    if hasattr(model_output, "model_dump"):
        return model_output.model_dump()
    return model_output.dict()


def _build_llm(
    model: str,
    api_key_name: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
    temperature: float = 0,
    timeout: int = 60,
    max_tokens: int = 8196,
    max_retries: int = 2,
):
    kwargs = {
        "model": model,
        "api_key": os.getenv(api_key_name),
        "temperature": temperature,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def base_llm_generator(
    llm,
    data,
    token_header,
    file_name_header,
    file_content_header,
    prompt,
    parser,
    verbose,
):
    """Evaluate prepared texts and collect parsed outputs."""
    if not data.get(file_content_header):
        raise ValueError("Input data is empty. Please provide valid file content.")

    evaluated_data = {token_header: data[token_header]} if token_header in data else {}

    with get_openai_callback() as cb:
        for text_index, text in enumerate(data[file_content_header]):
            file_name = data[file_name_header][text_index]
            if verbose:
                print(f"Processing index {text_index}: {file_name}")

            llm_message = ChatPromptTemplate.from_template(template=prompt).format_messages(
                intro_text=text,
                format_instructions=parser.get_format_instructions(),
            )

            try:
                start_time = time.time()
                response = llm.invoke(llm_message)
                output = _parse_model_output(parser, response.content)

                for key, value in _model_to_dict(output).items():
                    evaluated_data.setdefault(key, []).append(value)

                evaluated_data.setdefault("prompt_tokens", []).append(cb.prompt_tokens)
                evaluated_data.setdefault("total_tokens", []).append(cb.total_tokens)
                evaluated_data.setdefault("processing_time", []).append(time.time() - start_time)

                if verbose:
                    print(f"Parsed output:\n{output}")
            except Exception as exc:
                print(f"Error processing file {file_name}: {exc}")

    evaluated_data[file_content_header] = data[file_content_header]
    evaluated_data[file_name_header] = data[file_name_header]
    return evaluated_data


def prof_alborzi_congress_eval(
    data_path: str,
    prompt: str,
    parser,
    output_path: str = None,
    model: Optional[str] = "gpt-4.1-mini",
    temperature: Optional[float] = 0,
    max_retries: Optional[int] = 2,
    max_tokens: Optional[int] = 8196,
    timeout: Optional[int] = 60,
    verbose: Optional[bool] = False,
):
    """Evaluate congress abstracts one at a time."""
    llm = _build_llm(model, temperature=temperature, timeout=timeout, max_tokens=max_tokens, max_retries=max_retries)
    output_path = Path(output_path).resolve() if output_path else Path.cwd() / "LLM_outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    full_eval = []
    for abstract in data:
        abs_code = abstract.get("abstract_code", "")
        with get_openai_callback() as cb:
            if verbose:
                print(f"Processing Code {abs_code}:")

            llm_message = ChatPromptTemplate.from_template(template=prompt).format_messages(
                abs_text=abstract.get("Fulltext", ""),
                format_instructions=parser.get_format_instructions(),
            )

            try:
                start_time = time.time()
                response = llm.invoke(llm_message)
                output = _parse_model_output(parser, response.content)
                evaluated_data = {**abstract, **_model_to_dict(output)}
                evaluated_data["prompt_tokens"] = cb.prompt_tokens
                evaluated_data["total_tokens"] = cb.total_tokens
                evaluated_data["processing_time"] = time.time() - start_time
                full_eval.append(evaluated_data)
            except Exception as exc:
                print(f"Error processing Code {abs_code}: {exc}")

    output_file = output_path / f"LLM_Output_{model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_eval, f, indent=2, ensure_ascii=False)

    return full_eval


async def process_single_abstract(abstract, llm, prompt, parser, verbose: bool = False):
    """Evaluate one abstract asynchronously."""
    abs_code = abstract.get("abstract_code", "")
    try:
        with get_openai_callback() as cb:
            if verbose:
                print(f"Processing Code {abs_code}:")

            llm_message = ChatPromptTemplate.from_template(template=prompt).format_messages(
                abs_text=abstract.get("Fulltext", ""),
                format_instructions=parser.get_format_instructions(),
            )

            start_time = time.time()
            response = await llm.ainvoke(llm_message)
            output = _parse_model_output(parser, response.content)

            evaluated_data = {**abstract, **_model_to_dict(output)}
            evaluated_data["prompt_tokens"] = cb.prompt_tokens
            evaluated_data["total_tokens"] = cb.total_tokens
            evaluated_data["processing_time"] = time.time() - start_time

            if verbose:
                print(f"Parsed output:\n{output}")

            return evaluated_data
    except Exception as exc:
        print(f"Error processing Code {abs_code}: {exc}")
        return None


async def prof_alborzi_congress_eval_async(
    data_path: str,
    prompt: str,
    parser,
    output_path: str = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0,
    max_retries: int = 2,
    max_tokens: int = 8196,
    timeout: int = 60,
    verbose: bool = False,
    concurrency_limit: int = 15,
):
    """Evaluate congress abstracts concurrently."""
    llm = _build_llm(model, temperature=temperature, timeout=timeout, max_tokens=max_tokens, max_retries=max_retries)
    output_path = Path(output_path).resolve() if output_path else Path.cwd() / "LLM_outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def sem_task(abstract):
        async with semaphore:
            return await process_single_abstract(abstract, llm, prompt, parser, verbose)

    results = await asyncio.gather(*(sem_task(abstract) for abstract in data))
    full_eval = [result for result in results if result is not None]

    output_file = output_path / f"LLM_Output_{model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_eval, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"Finished processing {len(full_eval)} abstracts.")
        print(f"Output saved to {output_file}")

    return full_eval


@validate_call
def openai_response_generator(
    data: dict,
    prompt: str,
    parser,
    verbose: Optional[bool] = False,
    file_content_header: Optional[str] = "file_content",
    file_name_header: Optional[str] = "file_name",
    input_token_header: Optional[str] = "input_token_count",
    model: Optional[str] = "gpt-4.1-mini",
    api_key_name: Optional[str] = "OPENAI_API_KEY",
    temperature: Optional[float] = 0,
    timeout: Optional[int] = 60,
    max_tokens: Optional[int] = 8196,
    max_retries: Optional[int] = 2,
):
    """Run evaluation through OpenAI."""
    llm = _build_llm(
        model=model,
        api_key_name=api_key_name,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    return base_llm_generator(
        llm=llm,
        data=data,
        token_header=input_token_header,
        file_name_header=file_name_header,
        file_content_header=file_content_header,
        prompt=prompt,
        parser=parser,
        verbose=verbose,
    )


@validate_call
def openrouter_llm_response(
    data: dict,
    prompt,
    parser,
    model: Optional[
        Literal[
            "meta-llama/llama-4-maverick:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free",
        ]
    ] = "meta-llama/llama-3.3-70b-instruct:free",
    api_key_name: Optional[str] = "OPENROUTER_API_KEY",
    verbose: Optional[bool] = False,
    file_content_header: Optional[str] = "file_content",
    file_name_header: Optional[str] = "file_name",
    input_token_header: Optional[str] = "input_token_count",
    temperature: Optional[float] = 0,
    timeout: Optional[int] = 60,
    max_tokens: Optional[int] = 8196,
    max_retries: Optional[int] = 2,
):
    """Run evaluation through OpenRouter."""
    llm = _build_llm(
        model=model,
        api_key_name=api_key_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    return base_llm_generator(
        llm=llm,
        data=data,
        token_header=input_token_header,
        file_name_header=file_name_header,
        file_content_header=file_content_header,
        prompt=prompt,
        parser=parser,
        verbose=verbose,
    )
