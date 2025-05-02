from typing import Literal
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from src.prompt import rewrite_prompt_with_history


async def rewrite_prompt(
    query: str,
    history: list[str],
    model: str,
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Rewrite the given query based on the provided history using a language model.

    Args:
        query (str): The input query to be rewritten.
        history (list[ChatMessage]): A list of previous messages in the conversation.
        model (str): The language model to use for rewriting.
        prompt_mode (str): The mode of the system prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.

    Returns:
        str: The rewritten query.
    """
    assert prompt_mode in [
        "plain_text",
        "json",
    ], f"Invalid prompt_mode: {prompt_mode}. Must be one of ['plain_text', 'json']"

    llm = OpenAI(model=model)

    system_prompt = (
        rewrite_prompt_with_history.system_prompt.plain_text_version
        if prompt_mode == "plain_text"
        else rewrite_prompt_with_history.system_prompt.json_version
    )
    user_prompt = (
        rewrite_prompt_with_history.user_prompt.plain_text_version
        if prompt_mode == "plain_text"
        else rewrite_prompt_with_history.user_prompt.json_version
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=user_prompt.format(
                query=query, history="\n - ".join([msg for msg in history])
            ),
        ),
    ]

    response = await llm.achat(messages=messages)

    return response.message.content
