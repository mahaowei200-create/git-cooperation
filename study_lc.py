import os
import sqlite3
from tabnanny import check
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents import middleware
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import sqlite
from openai import OpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage,SystemMessage,HumanMessage,ToolMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain_tavily import TavilySearch



load_dotenv()
#模型创建
model = init_chat_model(
    model="qwen3.7-plus",
    model_provider=("openai"),
    api_key=os.getenv("OPENAI_API_KEY"),    
    base_url = os.getenv("OPENAI_BASE_URL")
)
#工具创建
web_search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"),max_results=5,topic="general")

#记忆创建
connection = sqlite3.connect("../resource/checkpoint.db",check_same_thread=False)
checkpointer = SqliteSaver(connection)
checkpointer.setup()

#中间件创建
summary_middleware = SummarizationMiddleware(
    model=ChatOpenAI(
        model="deepseek-chat",  
        base_url=os.getenv("DEEP_SEEK_BASE_URL"),
        api_key=os.getenv("DEEP_SEEK_API_KEY")
    ),
    trigger=("messages",20),
    keep=("messages",8)
)

#系统提示词创建
system_promt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""
#代理创建
agent = create_agent(
    model,
    tools=[web_search],
    system_prompt=system_promt,
    middleware=[summary_middleware],
    checkpointer=checkpointer
)

config = {"configurable":{"thread_id":"thread_6"}}

multimodal_message = HumanMessage(
    content=[
        {"type": "image",
         "url": "https://img.freepik.com/free-photo/arrangement-different-foods-organized-fridge_23-2149099882.jpg"},
        {"type": "text", "text": "帮我看看这些食材能做些什么？"}
    ])

response = agent.invoke({"messages": "我更喜欢第三道菜，你能详细的告诉我如何实现吗"}, config)

for message in response['messages']:
    message.pretty_print()


