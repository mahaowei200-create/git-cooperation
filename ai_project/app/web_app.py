import os

import streamlit as st

from dotenv import load_dotenv

from langchain.agents import create_agent

from langchain.tools import tool

from langchain_openai import ChatOpenAI



# ==========================================

# 0. 基础配置与环境变量

# ==========================================

load_dotenv()



# ==========================================

# 1. 网页框架配置

# ==========================================

st.set_page_config(page_title="AI 大管家", page_icon="🤖", layout="centered")

st.title("🤖 我的企业级 AI 管家")

st.caption("🚀 支持功能：查天气 | 算工资 | 翻阅本地硬核 PDF 知识库")



# ==========================================

# 2. 缓存大模型与知识库 (只在启动时加载一次！)

# ==========================================

@st.cache_resource

def init_agent():

    # 引入咱们手搓的极品齿轮

    import config.config as config

    from rag.document_parser import load_and_chunk_pdf

    from rag.retriever import HybridRetriever

   

    # 1. 加载本地 PDF 知识库

    chunks = load_and_chunk_pdf(config.PDF_PATH)

    engine = HybridRetriever(embed_model=config.EMBED_MODEL_NAME, rerank_model=config.RERANK_MODEL_NAME)

    engine.ingest_docs(chunks)

   

    # 2. 唤醒大模型

    llm = ChatOpenAI(model="deepseek-chat", temperature=0)

   

    # 3. 装备武器库

    @tool

    def search_weather(city: str) -> str:

        """查询指定城市的天气"""

        return {"北京": "雷阵雨，带伞", "上海": "晴，35度"}.get(city, "未知天气")



    @tool

    def calculate_entire_money(base: int, bonus: int) -> str:

        """计算总工资"""

        return f"总工资为：{base + bonus} 元"



    @tool

    def search_internal_knowledge(query: str) -> str:

        """查询公司内部 PDF 资料"""

        results = engine.search(query, top_k=5, final_k=3)

        context = "\n\n".join([doc for doc, score in results])

        return context if context else "未找到相关内容。"



    tools = [search_weather, calculate_entire_money, search_internal_knowledge]

   

    # 4. 组装并返回 Agent

    agent = create_agent(

        model=llm,

        tools=tools,

        system_prompt="你是一名强大的智能助理。如果用户需要查询规定、报销等，请调用知识库工具。用中文简短回答。"

    )

    return agent



# 获取管家实例

agent = init_agent()



# ==========================================

# 3. 绘制聊天界面与历史记忆

# ==========================================

# 在网页会话中初始化记忆列表

if "messages" not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": "老板你好！今天有什么吩咐？"}]



# 把历史聊天记录一条条画在网页上

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])



# ==========================================

# 4. 接收输入，呼叫 Agent

# ==========================================

# 这是一个漂亮的底部输入框

if user_input := st.chat_input("试着问问天气、算工资，或者 PDF 里的刁钻问题..."):

   

    # 1. 把老板的话画在右边

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):

        st.markdown(user_input)



    # 2. 管家开始思考并回答 (画在左边)

    with st.chat_message("assistant"):

        # 显示一个转圈圈的加载动画

        with st.spinner("🧠 管家正在思考并翻阅资料..."):

           

            # 瞬间调用 LangGraph 底层逻辑！

            result = agent.invoke(

                {"messages": [{"role": "user", "content": user_input}]}

            )

            final_answer = result["messages"][-1].content

           

            # 把最终答案打在屏幕上

            st.markdown(final_answer)

           

    # 3. 把管家的回答存入记忆

    st.session_state.messages.append({"role": "assistant", "content": final_answer})