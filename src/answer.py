from typing import Literal
from vllm import LLM, SamplingParams
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.utils import config
from src.prompt import (
    openai_answer_prompt,
    vllm_answer_prompt,
    get_prompt,
    gen_data_prompt,
)


def load_model(model_path: str, **kwargs) -> LLM:
    return LLM(model=model_path, **kwargs)


vllm_model = None
if config.llm_options.provider == "vllm":
    vllm_model = load_model(config.llm_options.model, **config.llm_options.kwargs)


async def vllm_generate_answer(
    query: str,
    contexts: list[str],
    max_tokens: int = 1024,
    prompt_mode: Literal[
        "plain_text", "json", "xml", "markdown", "yaml"
    ] = "plain_text",
) -> str:
    """
    Generate an answer to the given query based on the provided contexts.
    Args:
        query (str): The input query.
        contexts (list[str]): A list of context strings to use for generating the answer.
        model (str): The model to use for generation.
        prompt_mode (str): The mode of the system prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.
            - "xml": Use the XML version of the system prompt.
            - "markdown": Use the Markdown version of the system prompt.
            - "yaml": Use the YAML version of the system prompt.
    Returns:
        str: The generated answer.
    """
    assert vllm_model is not None, "VLLM model is not loaded."

    assert prompt_mode in [
        "plain_text",
        "json",
        "xml",
        "markdown",
        "yaml",
    ], f"Invalid prompt_mode: {prompt_mode}. Must be one of ['plain_text', 'json', 'xml', 'markdown', 'yaml']"

    system_prompt, user_prompt = get_prompt(prompt=gen_data_prompt, mode=prompt_mode)

    final_context = ""
    for context in contexts:
        final_context += "<DOCUMENT>" + context + "</DOCUMENT>\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt.format(query=query, context=final_context),
        },
    ]

    response = vllm_model.chat(
        messages=messages,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    return response[0].outputs[0].text


async def openai_generate_answer(
    query: str,
    contexts: list[str],
    model: str,
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Generate an answer to the given query based on the provided contexts.

    Args:
        query (str): The input query.
        contexts (list[str]): A list of context strings to use for generating the answer.

    Returns:
        str: The generated answer.
    """
    assert prompt_mode in [
        "plain_text",
        "json",
    ], f"Invalid prompt_mode: {prompt_mode}. Must be one of ['plain_text', 'json']"

    llm = OpenAI(model=model)

    system_prompt, user_prompt = get_prompt(
        prompt=openai_answer_prompt, mode=prompt_mode
    )
    final_context = "\n=========\n".join(contexts)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=user_prompt.format(query=query, final_context=final_context),
        ),
    ]

    response = await llm.achat(messages=messages)

    return response.message.content
