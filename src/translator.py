import asyncio
from loguru import logger
from typing import Literal
from googletrans import Translator
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.prompt import translate_prompt


async def detect_lang(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text whose language is to be detected.

    Returns:
        str: The detected language code (e.g., 'en' for English).
    """
    async with Translator() as translator:
        result = await translator.detect(text)
        return result.lang


async def translate_ggtrans(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate the given text using Google Translate.

    Args:
        text (str): The text to be translated.
        src_lang (str): The source language code (e.g., 'en' for English).
        tgt_lang (str): The target language code (e.g., 'vi' for Vietnamese).

    Returns:
        str: The translated text.
    """
    async with Translator() as translator:
        result = await translator.translate(text, src=src_lang, dest=tgt_lang)
        return result.text


async def translate_llm(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model: str,
    system_prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Translate the given text using a language model.

    Args:
        text (str): The text to be translated.
        src_lang (str): The source language code (e.g., 'en' for English).
        tgt_lang (str): The target language code (e.g., 'vi' for Vietnamese).
        model (str): The language model to use for translation.
        system_prompt_mode (str): The mode of the system prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.

    Returns:
        str: The translated text.
    """
    llm = OpenAI(model=model)

    system_prompt = (
        translate_prompt.system_prompt.plain_text_version
        if system_prompt_mode == "plain_text"
        else translate_prompt.system_prompt.json_version
    )

    def get_lang(lang: str) -> str:
        if lang == "en":
            return "English"
        elif lang == "vi":
            return "Vietnamese"
        return lang

    messages = [
        ChatMessage(
            role="system",
            content=system_prompt.format(
                src_lang=get_lang(src_lang), tgt_lang=get_lang(tgt_lang)
            ),
        ),
        ChatMessage(
            role="user", content=translate_prompt.user_prompt.format(query=text)
        ),
    ]
    response = await llm.achat(messages=messages)

    return response.message.content


async def translate(
    *,
    text: str,
    tgt_lang: str,
    method: Literal["ggtrans", "llm"],
    model: str | None = None,
    system_prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Translate the given text from source language to target language.

    Args:
        text (str): The text to be translated.
        tgt_lang (str): The target language code (e.g., 'vi' for Vietnamese).
        method (Literal["ggtrans", "llm"]): The translation method to use.
            - "ggtrans": Use Google Translate.
            - "llm": Use a language model for translation.
        model (str | None): The OpenAI's LLM to use for translation (if method is "llm").

    Returns:
        str: The translated text.
    """
    text_lang = await detect_lang(text)
    if text_lang == tgt_lang:
        return text

    assert method in [
        "ggtrans",
        "llm",
    ], "Invalid translation method. Use 'ggtrans' or 'llm'."

    logger.debug(f"Using: {method}")

    if method == "ggtrans":
        return await translate_ggtrans(text, text_lang, tgt_lang)

    elif method == "llm":
        assert model is not None, "Model must be specified when using 'llm' method."
        return await translate_llm(text, text_lang, tgt_lang, model, system_prompt_mode)


async def translate_final_answer(
    *,
    query: str,
    answer: str,
    method: Literal["ggtrans", "llm"],
    model: str | None = None,
    system_prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Translate the final answer to the language of the query.
    Args:
        query (str): The input query.
        answer (str): The answer to be translated.
        method (Literal["ggtrans", "llm"]): The translation method to use.
            - "ggtrans": Use Google Translate.
            - "llm": Use a language model for translation.
        model (str | None): The OpenAI's LLM to use for translation (if method is "llm").
        system_prompt_mode (str): The mode of the system prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.

    Returns:
        str: The translated answer.
    """
    assert method in [
        "ggtrans",
        "llm",
    ], "Invalid translation method. Use 'ggtrans' or 'llm'."

    query_lang, answer_lang = await asyncio.gather(
        detect_lang(query),
        detect_lang(answer),
    )

    if query_lang == answer_lang:
        return answer

    logger.debug(f"Translate: {answer_lang} -> {query_lang}")

    return await translate(
        text=answer,
        tgt_lang=query_lang,
        method=method,
        model=model,
        system_prompt_mode=system_prompt_mode,
    )
