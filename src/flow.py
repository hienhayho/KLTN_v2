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
from src.rewrite_prompt import rewrite_prompt
from src.translator import translate, translate_final_answer
from src.answer import (
    openai_generate_answer,
    vllm_generate_answer,
    openai_answer_from_history,
)
from src.constants import (
    out_domain_responses,
    greeting_responses,
    bye_responses,
    not_supported_languages_responses,
)


class PreprocessEvent(Event):
    query: str
    history: list[dict]


class RetrieveEvent(Event):
    query: str


class AfterRetrieveEvent(Event):
    contexts: list[str]


class AnswerEvent(Event):
    contexts: list[str] = []
    answer: str


class FinalAnswerEvent(StopEvent):
    answer: str
    final_query: str
    contexts: list[str]


class AppFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> PreprocessEvent:
        logger.info("query: {}", ev.query)

        # Store the query in the context to use later
        await ctx.set("query", ev.query)
        await ctx.set("only_retrieve", ev.only_retrieve)

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
        translated_query, text_lang = await translate(
            text=ev.query,
            tgt_lang="vi",
            method=config.translate_options.method,
            model=config.translate_options.model,
            prompt_mode=config.translate_options.prompt_mode,
        )
        if text_lang not in ["vi", "en"]:
            random_response = random.choice(not_supported_languages_responses)
            return AnswerEvent(answer=random_response)

        logger.info(f"Translated query: {translated_query}")

        # Rewrite the query based on the history
        history = ev.history
        translated_query = await rewrite_prompt(
            query=translated_query,
            history=history if history else [],
            model=config.rewrite_options.model,
            prompt_mode=config.rewrite_options.prompt_mode,
        )
        logger.info(f"Rewritten query: {translated_query}")

        await ctx.set("final_query", translated_query)

        # Get topic from the query
        topic = await check_domain(
            text=translated_query,
            model=config.check_domain_options.model,
            prompt_mode=config.check_domain_options.prompt_mode,
        )
        # topic = "pass"

        logger.info(f"Topic: {topic}")
        if topic == "other":
            random_response = random.choice(out_domain_responses)
            return AnswerEvent(answer=random_response)

        elif topic == "greeting":
            random_response = random.choice(greeting_responses)
            return AnswerEvent(answer=random_response)

        elif topic == "bye":
            random_response = random.choice(bye_responses)
            return AnswerEvent(answer=random_response)

        # Answer from history if available
        if ev.history:
            logger.info("History found, try generating answer from history ...")
            answer = await openai_answer_from_history(
                model="gpt-4.1-mini",
                query=ev.query,
                history=ev.history,
            )
            if answer:
                logger.info(f"Answer from history: {answer}")
                return FinalAnswerEvent(
                    answer=answer, final_query=ev.query, contexts=[]
                )

        # Otherwise, proceed to the retrieval step
        return RetrieveEvent(query=translated_query)

    @step
    async def retrieve(
        self, ctx: Context, ev: RetrieveEvent
    ) -> AfterRetrieveEvent | FinalAnswerEvent:
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

        only_retrieve = await ctx.get("only_retrieve")
        if only_retrieve:
            # If only retrieve is true, return the contexts
            return FinalAnswerEvent(answer="", final_query=ev.query, contexts=contexts)

        return AfterRetrieveEvent(contexts=contexts)

    @step
    async def pre_answer(self, ctx: Context, ev: AfterRetrieveEvent) -> AnswerEvent:
        """
        Generate an answer based on the retrieved contexts.

        Args:
            contexts (list): The retrieved contexts.

        Returns:
            str: The generated answer.
        """
        # Generate the answer
        query = await ctx.get("final_query")

        # Check provider and generate answer
        if config.llm_options.provider == "openai":
            answer = await openai_generate_answer(
                query=query,
                contexts=ev.contexts,
                model=config.llm_options.model,
                prompt_mode=config.llm_options.prompt_mode,
            )
            return AnswerEvent(answer=answer, contexts=ev.contexts)
        elif config.llm_options.provider == "vllm":
            answer = await vllm_generate_answer(
                query=query,
                contexts=ev.contexts,
                max_tokens=config.llm_options.inference_options.max_tokens,
                prompt_mode=config.llm_options.prompt_mode,
            )
            return AnswerEvent(answer=answer, contexts=ev.contexts)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {config.llm_options.provider}. "
                "Supported providers are: ['openai', 'vllm']"
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

        answer, _ = await translate_final_answer(
            query=query,
            answer=ev.answer,
            method=config.translate_options.method,
            model=config.translate_options.model,
            prompt_mode=config.translate_options.prompt_mode,
        )
        answer = answer.strip().replace(".docx", "")
        logger.info(f"Final answer: {answer}")

        return FinalAnswerEvent(
            answer=answer,
            final_query=await ctx.get("final_query", default=query),
            contexts=ev.contexts,
        )
