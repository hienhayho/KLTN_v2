import chainlit as cl
from loguru import logger
from llama_index.core.llms import ChatMessage

from src.flow import AppFlow


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


@cl.on_message
async def run(message: cl.Message):
    msg = cl.Message(content="", author="Assistant")
    history = cl.user_session.get("history")
    user_history = [msg.content for msg in history if msg.role == "user"]

    logger.debug(f"history: {'\n- '.join(user_history)}")

    flow = AppFlow(timeout=1000, verbose=False)

    res = await flow.run(query=message.content, history=user_history)

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
