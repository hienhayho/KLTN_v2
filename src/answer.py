from typing import Literal
from json_repair import loads
from vllm import LLM, SamplingParams
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.utils import config
from src.prompt import (
    openai_answer_prompt,
    get_prompt,
    gen_data_prompt,
    system_prompt_answer_from_history,
    user_prompt_answer_from_history,
)


def load_model(model_path: str, **kwargs) -> LLM:
    return LLM(model=model_path, **kwargs)


vllm_model = None
if config.llm_options.provider == "vllm":
    vllm_model = load_model(config.llm_options.model, **config.llm_options.kwargs)


async def openai_answer_from_history(
    model: str,
    query: str,
    history: list[dict],
):
    """
    Generate an answer based on the input query and history using OpenAI.

    Args:
        model (str): The OpenAI model to use for generation.
        query (str): The input query.
        history (list[dict]): A list of previous messages in the conversation.

    Returns:
        str: The generated answer.
    """
    llm = OpenAI(model=model, temperature=0.0)

    history_str = ""
    for message in history:
        if message["role"] == "user":
            history_str += f"<USER>: {message['content']}\n"
        elif message["role"] == "assistant":
            history_str += f"<ASSISTANT>: {message['content']}\n"

    print(user_prompt_answer_from_history.format(query=query, history=history_str))

    messages = [
        ChatMessage(role="system", content=system_prompt_answer_from_history),
        ChatMessage(
            role="user",
            content=user_prompt_answer_from_history.format(
                query=query, history=history_str
            ),
        ),
    ]

    response = await llm.achat(
        messages=messages, response_format={"type": "json_object"}
    )

    return loads(response.message.content)["answer"]


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

    system_prompt, user_prompt = get_prompt(
        prompt=gen_data_prompt, mode=prompt_mode, is_cot=False
    )

    system_prompt = """Persona\n- You are a reasoning assistant. Your task is to provide logical reasoning to answer the given question based on provided context.\n\nInstruction:\n- Given the question, context and answer above, provide a logical reasoning for that answer.\n- Please use the format of: <REASON>: {{reason}} <ANSWER>: {{answer}}"""

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
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    return response[0].outputs[0].text


async def openai_generate_answer(
    query: str,
    contexts: list[str],
    model: str,
    prompt_mode: Literal[
        "plain_text", "json", "xml", "markdown", "yaml"
    ] = "plain_text",
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
        "xml",
        "markdown",
        "yaml",
    ], f"Invalid prompt_mode: {prompt_mode}"

    llm = OpenAI(model=model, temperature=0.0)

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
