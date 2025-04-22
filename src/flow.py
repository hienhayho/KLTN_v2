import random
from loguru import logger
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    Workflow,
    StopEvent,
    StartEvent,
)


from src.utils import config
from src.retrieve import retrive
from src.check_domain import check_domain
from src.answer import openai_generate_answer
from src.rewrite_prompt import rewrite_prompt
from src.constants import out_domain_responses
from src.translator import translate, translate_final_answer


class PreprocessEvent(Event):
    query: str
    history: list[str]


class RetrieveEvent(Event):
    query: str


class PreAnswerEvent(Event):
    contexts: list[str]


class AnswerEvent(Event):
    answer: str


class FinalAnswerEvent(StopEvent):
    answer: str
    final_query: str


class AppFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> PreprocessEvent:
        logger.info("query: {}", ev.query)

        # Store the query in the context to use later
        await ctx.set("query", ev.query)

        # Trigger the preprocessing step
        return PreprocessEvent(query=ev.query, history=ev.history)

    @step
    async def preprocessing_query(
        self, ctx: Context, ev: PreprocessEvent
    ) -> AnswerEvent | RetrieveEvent:
        """
        Preprocess the input query by stripping leading and trailing whitespace.

        Args:
            query (str): The input query.

        Returns:
            str: The preprocessed query.
        """
        # Translate the query to Vietnamese if needed
        translated_query = await translate(
            text=ev.query,
            tgt_lang="vi",
            method=config.translate_options.method,
            model=config.translate_options.model,
            system_prompt_mode=config.translate_options.system_prompt_mode,
        )
        logger.info(f"Translated query: {translated_query}")

        # Rewrite the query based on the history
        history = ev.history
        if history:
            translated_query = await rewrite_prompt(
                query=translated_query,
                history=history,
                model=config.rewrite_options.model,
                system_prompt_mode=config.rewrite_options.system_prompt_mode,
            )
            logger.info(f"Rewritten query: {translated_query}")

        await ctx.set("final_query", translated_query)
        # Check in-domain query
        is_in_domain = await check_domain(
            text=translated_query,
            model=config.check_domain_options.model,
            system_prompt_mode=config.check_domain_options.system_prompt_mode,
        )

        if not is_in_domain:
            logger.debug("Query is out of domain, returning out-of-domain response.")
            response = random.choice(out_domain_responses)
            return AnswerEvent(answer=response)

        # Otherwise, proceed to the retrieval step
        return RetrieveEvent(query=translated_query)

    @step
    async def retrieve(self, ev: RetrieveEvent) -> PreAnswerEvent:
        """
        Retrieve the answer based on the preprocessed query.

        Args:
            query (str): The preprocessed query.

        Returns:
            str: The retrieved answer.
        """
        logger.info(f"Retrieving answer for query: {ev.query}")

        # Retrive phase
        contexts = await retrive(
            query=ev.query,
            top_k=config.retriever_options.top_k,
            method=config.retriever_options.method,
        )

        return PreAnswerEvent(contexts=contexts)

    @step
    async def pre_answer(self, ctx: Context, ev: PreAnswerEvent) -> AnswerEvent:
        """
        Generate an answer based on the retrieved contexts.

        Args:
            contexts (list): The retrieved contexts.

        Returns:
            str: The generated answer.
        """
        # Generate the answer
        query = await ctx.get("query")

        # Check provider and generate answer
        if config.llm_options.provider == "openai":
            answer = await openai_generate_answer(
                query=query,
                contexts=ev.contexts,
                model=config.llm_options.model,
                system_prompt_mode=config.llm_options.system_prompt_mode,
            )
            return AnswerEvent(answer=answer)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {config.llm_options.provider}. "
                "Supported providers are: ['openai']"
            )

    @step
    async def final_answer(self, ctx: Context, ev: AnswerEvent) -> FinalAnswerEvent:
        """
        Generate an answer based on the input query.

        Args:
            query (str): The input query.

        Returns:
            str: The generated answer.
        """
        # Generate the answer
        logger.debug(f"Get initial answer: {ev.answer}")

        # Translate the answer to the language of the query
        query = await ctx.get("query")

        answer = await translate_final_answer(
            query=query,
            answer=ev.answer,
            method=config.translate_options.method,
            model=config.translate_options.model,
            system_prompt_mode=config.translate_options.system_prompt_mode,
        )
        answer = answer.strip().replace(".docx", "")
        logger.info(f"Final answer: {answer}")

        return FinalAnswerEvent(answer=answer, final_query=await ctx.get("final_query"))
