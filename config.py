bm_25_index = "contextual_chunk_with_context_db/bm25_retriever"
vector_db_persist_dir = "contextual_chunk_with_context_db/chroma_db"
embed_model_name = "text-embedding-3-large"
semantic_retriever_weight = 0.8
bm25_retriever_weight = 0.2

detect_language_options = dict(
    method="lingua",  # ["ggtrans", "lingua"]
)
translate_options = dict(method="llm", model="gpt-4.1-mini", prompt_mode="json")
rewrite_options = dict(
    model="gpt-4.1-mini",
    prompt_mode="json",
)
check_domain_options = dict(
    model="gpt-4.1-mini",
    prompt_mode="json",
)
retriever_options = dict(
    method="hybrid",
    top_k=5,
)
llm_options = dict(
    provider="vllm",
    # model="/mlcv3/WorkingSpace/Personal/baotg/xai_after/LLaMA-Factory/output/llama3-3b_5_contexts/",
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/LLaMA-Factory/output/qwen2_5_3b_5_contexts",
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/kltn_seallm_1.5b",
    # ====================================================
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/sft_new_data_models/seallm_1_5B_new_data_json",
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/sft_new_data_models/seallm_1_5B_new_data_markdown",
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/sft_new_data_models/seallm_1_5B_new_data_plain_text",
    # model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/sft_new_data_models/seallm_1_5B_new_data_xml",
    model="/mlcv2/WorkingSpace/Personal/hienht/KLTN_new/sft_new_data_models/qwen_2_5_3B_new_data_yaml",
    prompt_mode="yaml",
    kwargs=dict(
        gpu_memory_utilization=0.9,
        max_model_len=16384,
    ),
    inference_options=dict(max_tokens=2048),
)
# llm_options = dict(
#     provider="openai",
#     model="gpt-4.1-mini",
#     prompt_mode="json",
# )
llm_as_a_judge_options = dict(
    model="gpt-4.1-mini",
    prompt_mode="json",
)
