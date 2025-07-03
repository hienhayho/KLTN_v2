import chainlit as cl
from typing import Dict, Optional
from loguru import logger
from chainlit.types import ThreadDict
from llama_index.core.llms import ChatMessage

from src.flow import AppFlow


# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#     return default_user


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Hỏi thông tin về dịch vụ công",
            message="Xin thông tin các biểu mẫu",
        ),
        cl.Starter(
            label="Hỏi ngoài lề",
            message="Hôm nay ngày mấy ?",
        ),
        cl.Starter(label="Chào hỏi", message="Hello bạn =))"),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    new_memory = []
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    # CODE HERE
    for message in root_messages:
        if message["output"]:
            if message["type"] == "user_message":
                new_memory.append(ChatMessage(role="user", content=message["output"]))
            else:
                new_memory.append(
                    ChatMessage(role="assistant", content=message["output"])
                )

    cl.user_session.set("chat_messages", new_memory)


@cl.on_message
async def run(message: cl.Message):
    msg = cl.Message(content="", author="Assistant")
    history = cl.user_session.get("history")
    user_history = [{
        "role": msg.role,
        "content": msg.content} for msg in history]

    # logger.debug(f"history: {'\n- '.join(user_history)}")

    flow = AppFlow(timeout=1000, verbose=False)

    res = await flow.run(
        query=message.content, history=user_history, only_retrieve=False
    )

    response = ""

    for token in res.answer:
        response += token + ""
        await msg.stream_token(token)

    history.append(
        ChatMessage(
            role="user",
            content=res.final_query,
        )
    )
    history.append(
        ChatMessage(
            role="assistant",
            content=response,
        )
    )

    cl.user_session.set("history", history)

    await msg.send()
