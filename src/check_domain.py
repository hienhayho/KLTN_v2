from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.prompt import check_domain_prompt

load_dotenv()


class CheckDomainResponse(BaseModel):
    topic: Literal["greeting", "bye", "administration", "other"]


async def check_domain(
    *,
    text: str,
    model: str,
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Check if the given text is in-domain or out-of-domain using a language model.

    Args:
        text (str): The text to be checked.
        model (str): The language model to use for checking.
        prompt_mode (str): The mode of the system prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.

    Returns:
        str: The response indicating whether the text is in-domain or out-of-domain.
    """
    llm = OpenAI(model=model)

    assert prompt_mode in ["plain_text", "json"], (
        f"Invalid prompt_mode: {prompt_mode}. " "Must be one of ['plain_text', 'json']"
    )

    system_prompt = (
        check_domain_prompt.system_prompt.plain_text_version
        if prompt_mode == "plain_text"
        else check_domain_prompt.system_prompt.json_version
    )
    user_prompt = (
        check_domain_prompt.user_prompt.plain_text_version
        if prompt_mode == "plain_text"
        else check_domain_prompt.user_prompt.json_version
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt.format(query=text)),
    ]

    response = await llm.achat(messages=messages)
    return CheckDomainResponse.model_validate_json(response.message.content).topic
