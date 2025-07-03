import asyncio
import logging
import pycountry
from loguru import logger
from typing import Literal
from googletrans import Translator
from lingua import Language, LanguageDetectorBuilder
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    after_log,
    before_sleep_log,
    retry_if_exception_type,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.utils import config
from src.prompt import translate_prompt, get_prompt


async def async_detect_lang(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): The text whose language is to be detected.
        method (str): The method to use for language detection.

    Returns:
        str: The detected language code (e.g., 'en' for English).
    """
    method = config.detect_language_options.method

    if method == "ggtrans":
        return await ggtrans_detect_lang(text)
    elif method == "lingua":
        return lingua_detect_lang(text)
    else:
        raise ValueError(f"Invalid method: {method}")


async def ggtrans_detect_lang(text: str):
    """
    Detect the language of the given text using Google Translate.
    """
    small_text = text.split("\n")[:2]

    async with Translator() as translator:
        result = await translator.detect("\n".join(small_text))
        return result.lang


def lingua_detect_lang(
    text: str,
) -> str:
    """
    Detect the language of the given text using lingua.

    Args:
        text (str): The text whose language is to be detected.
        src_lang (str): The source language code (e.g., 'en' for English).

    Returns:
        str: The detected language code (e.g., 'en' for English).
    """
    languages = [Language.ENGLISH, Language.VIETNAMESE, Language.JAPANESE]
    # languages = [Language.ENGLISH, Language.VIETNAMESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    # detector = LanguageDetectorBuilder.from_all_languages().build()
    language = detector.detect_language_of(text)

    return language.iso_code_639_1.name.lower() if language else "unknown"


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
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> str:
    """
    Translate the given text using a language model.

    Args:
        text (str): The text to be translated.
        src_lang (str): The source language code (e.g., 'en' for English).
        tgt_lang (str): The target language code (e.g., 'vi' for Vietnamese).
        model (str): The language model to use for translation.
        prompt_mode (str): The mode of the prompt to use.
            - "plain_text": Use the plain text version of the system prompt.
            - "json": Use the JSON version of the system prompt.

    Returns:
        str: The translated text.
    """
    llm = OpenAI(model=model)

    system_prompt, user_prompt = get_prompt(prompt=translate_prompt, mode=prompt_mode)

    def get_lang(iso_code: str) -> str:
        lang = pycountry.languages.get(alpha_2=iso_code.lower())
        return lang.name.lower() if lang else iso_code

    messages = [
        ChatMessage(
            role="system",
            content=system_prompt.format(
                src_lang=get_lang(src_lang), tgt_lang=get_lang(tgt_lang)
            ),
        ),
        ChatMessage(role="user", content=user_prompt.format(query=text)),
    ]
    response = await llm.achat(messages=messages)

    return response.message.content


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_fixed(5),
    after=after_log(logger, logging.DEBUG),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    retry=retry_if_exception_type(BaseException),
)
async def translate(
    *,
    text: str,
    tgt_lang: str,
    method: Literal["ggtrans", "llm"],
    model: str | None = None,
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> tuple[str, str]:
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
        tuple[str, str]: The translated text and the source language code.
    """
    text_lang = await async_detect_lang(text)
    if text_lang == tgt_lang:
        return text, text_lang

    if text_lang not in ["vi", "en"]:
        logger.warning(
            f"Detected language '{text_lang}' is not supported for translation. "
            "Only 'vi' (Vietnamese) and 'en' (English) are supported."
        )
        return "", text_lang

    assert method in [
        "ggtrans",
        "llm",
    ], "Invalid translation method. Use 'ggtrans' or 'llm'."

    logger.debug(f"Using: {method}")

    if method == "ggtrans":
        return await translate_ggtrans(text, text_lang, tgt_lang), text_lang

    elif method == "llm":
        assert model is not None, "Model must be specified when using 'llm' method."
        return (
            await translate_llm(text, text_lang, tgt_lang, model, prompt_mode),
            text_lang,
        )


async def translate_final_answer(
    *,
    query: str,
    answer: str,
    method: Literal["ggtrans", "llm"],
    model: str | None = None,
    prompt_mode: Literal["plain_text", "json"] = "plain_text",
) -> tuple[str, str]:
    """
    Translate the final answer to the language of the query.
    Args:
        query (str): The input query.
        answer (str): The answer to be translated.
        method (Literal["ggtrans", "llm"]): The translation method to use.
            - "ggtrans": Use Google Translate.
            - "llm": Use a language model for translation.
        model (str | None): The OpenAI's LLM to use for translation (if method is "llm").
        prompt_mode (str): The mode of the system prompt to use.
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
        async_detect_lang(query),
        async_detect_lang(answer),
    )

    if query_lang == answer_lang:
        return answer, query_lang

    logger.debug(f"Translate: {answer_lang} -> {query_lang}")

    return await translate(
        text=answer,
        tgt_lang=query_lang,
        method=method,
        model=model,
        prompt_mode=prompt_mode,
    )
