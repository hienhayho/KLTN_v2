from typing import Literal
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.prompt import openai_answer_prompt


async def openai_generate_answer(
    query: str,
    contexts: list[str],
    model: str,
    system_prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Generate an answer to the given query based on the provided contexts.

    Args:
        query (str): The input query.
        contexts (list[str]): A list of context strings to use for generating the answer.

    Returns:
        str: The generated answer.
    """
    assert system_prompt_mode in [
        "plain_text",
        "json",
    ], f"Invalid system_prompt_mode: {system_prompt_mode}. Must be one of ['plain_text', 'json']"

    llm = OpenAI(model=model)

    system_prompt = (
        openai_answer_prompt.system_prompt.plain_text_version
        if system_prompt_mode == "plain_text"
        else openai_answer_prompt.system_prompt.json_version
    )

    final_context = "\n=========\n".join(contexts)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=openai_answer_prompt.user_prompt.format(
                query=query, final_context=final_context
            ),
        ),
    ]

    response = await llm.achat(messages=messages)

    return response.message.content
