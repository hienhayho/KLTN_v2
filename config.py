bm_25_index = "./bm25_retriever_with_context"
vector_db_collection = "output_large_hybrid_with_context"
embed_model_name = "text-embedding-3-large"
semantic_retriever_weight = 0.8
bm25_retriever_weight = 0.2

translate_options = dict(
    method="llm", model="gpt-4.1-mini", system_prompt_mode="plain_text"
)
rewrite_options = dict(
    model="gpt-4.1-mini",
    system_prompt_mode="plain_text",
)
check_domain_options = dict(
    model="gpt-4.1-mini",
    system_prompt_mode="plain_text",
)
retriever_options = dict(
    method="hybrid",
    top_k=8,
)
llm_options = dict(
    provider="openai",
    model="gpt-4.1-mini",
    system_prompt_mode="plain_text",
)
