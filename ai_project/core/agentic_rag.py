import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

# ==========================================
# 0. 基础配置与环境封印
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

# 使用 ChatOpenAI 桥接 DeepSeek API
llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# ==========================================
# 1. 初始化物理引擎 (全局加载一次)
# ==========================================
import config.config as config
from rag.document_parser import load_and_chunk_pdf
from rag.retriever import HybridRetriever

print("📚 正在加载本地知识库，请稍候...")
chunks = load_and_chunk_pdf(config.PDF_PATH)
engine = HybridRetriever(embed_model=config.EMBED_MODEL_NAME, rerank_model=config.RERANK_MODEL_NAME)
engine.ingest_docs(chunks)
print("✅ 知识库就绪！\n" + "="*40)

# ==========================================
# 2. 定义 State 与 强约束的 Pydantic 输出结构
# ==========================================
class GraphState(TypedDict):
    question: str      # 用户的当前问题
    documents: str     # 检索到的文本片段拼接
    answer: str        # 大模型生成的回答
    retries: int       # 记录重试次数，防止死循环
    history:str
    summary: str   # 🆕 新增
    tool: str
    memory: str   # ⭐ B2新增
    memory_context: str

# 💡 裁判 1 的结构：评估资料是否相关
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="文档是否和问题相关, 只能输出 'yes' 或 'no'")

# 💡 裁判 2 的结构：评估生成内容是否有幻觉、是否切题
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="回答是否完全基于资料且回答了问题, 只能输出 'yes' 或 'no'")


# ==========================================
# 3. 定义 Nodes (打工人小队)
# ==========================================

# 🧑‍💻 节点 1：检索员
def retrieve(state: GraphState):
    print("🔍 [节点: retrieve] 正在底层知识库中翻阅资料...")
    question = state["question"]
    history= state["history"]
    memory_context = state.get("memory_context", "")
    # 调用你的双路检索引擎
    query = f"""
用户问题：{question}

相关记忆：
{memory_context}
"""
    results = engine.search(query, top_k=5, final_k=3)
    context = "\n\n".join([doc for doc, score in results])
    
    return {
    "documents": context,
    "question": question,
    "memory_context": memory_context   # ⭐ 必须传下去
}


# 🧑‍💻 节点 2：生成器
def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    history = state.get("history", "")
    summary = state.get("summary", "")
    memory = state.get("memory", "")   # ⭐ 新增
    memory_context = state.get("memory_context", "")

    prompt = f"""
你是一个拥有长期记忆的AI助手。

【相关记忆（Tool Memory）】
{memory_context}

【长期摘要】
{summary}

【短期对话】
{history}

【知识库】
{documents}

【问题】
{question}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def tool_router(state: GraphState):
    question = state["question"]

    prompt = f"""
你是AI工具调度器（非常重要）

你必须判断用户问题属于：

【memory】
- 用户偏好
- 个人经历
- 之前说过的事情
- 对话历史

【rag】
- PDF / 文档 / 知识库内容

【llm】
- 常识问题
- 闲聊
- 不需要记忆也不需要文档

输出严格 JSON：
{"tool": "memory" | "rag" | "llm"}

问题：
{question}
"""

    response = llm.invoke(prompt)

    import json, re

    try:
        match = re.search(r"\{.*\}", response.content)
        tool = json.loads(match.group())["tool"]
    except:
        tool = "llm"

    tool = tool.lower()

    print("🧠 TOOL DECISION:", tool)

    return {"tool": tool}


def generate_direct(state):
    question = state["question"]
    memory = state.get("memory", "")
    summary = state.get("summary", "")

    prompt = f"""
你是AI助手。

【语义记忆】
{memory}

【摘要记忆】
{summary}

问题：{question}
"""

    return {"answer": llm.invoke(prompt).content}


# 🧑‍💻 节点 3：问题改写员
def rewrite_question(state: GraphState):
    print("🔄 [节点: rewrite] 流程触发打回！正在重新优化搜索关键词...")
    question = state["question"]
    retries = state.get("retries", 0) + 1
    
    prompt = f"用户的问题是：【{question}】。\n当前搜索遇到的瓶颈。请你换一种更专业、更宽泛的术语表达方式重新提问，以便更容易在文档中搜到结果。只输出改写后的新问题即可。"
    response = llm.invoke(prompt)
    
    return {
    "question": response.content,
    "retries": retries,
    "history": state.get("history", "")
}
# ==========================================
# 4. 定义 Conditional Edges (核心路由器)
# ==========================================

# ⚖️ 路由 1：初审裁判（检查检索到的资料质量）
def grade_documents_route(state: GraphState) -> str:
    print("⚖️ [裁判路由] 正在审查检索到的资料质量...")
    
    if state.get("retries", 0) >= 2:
        print("⚠️ [裁判路由] 重试次数已达上限，强行终止检索，交付生成组！")
        return "generate"

    question = state["question"]
    documents = state["documents"]
    
    # 【修改部分】：改用更纯粹的 JsonOutputParser
    parser = JsonOutputParser()
    
    prompt = f"""你是一个严格的论文初审专家。请判断以下检索到的【资料】是否包含可以回答用户【问题】的核心事实信息。
    
【输出格式要求】：
必须输出一个标准的 JSON 字符串，包含一个 key 叫 "binary_score"，其值只能是 "yes" 或 "no"。
绝对不要输出任何其他解释性文字或整个 Schema 结构。
示例：{{"binary_score": "yes"}}

资料：{documents}
问题：{question}"""

    response = llm.invoke(prompt)
    
    try:
        # 解析为普通的 Python 字典
        score_dict = parser.invoke(response)
        grade = score_dict.get("binary_score", "yes")
    except Exception as e:
        print(f"⚠️ [裁判路由] 解析异常，默认放行。错误：{e}")
        grade = "yes"
    
    if str(grade).lower() == "yes":
        print("✅ [裁判路由] 资料合格！批准放行到生成组！")
        return "generate"
    else:
        print("❌ [裁判路由] 资料不相关！打回改写组！")
        return "rewrite_question"

def memory_node(state):
    question = state["question"]
    memory = state["memory"]

    memory_context = memory.search(question)   # 向量记忆
    summary = state.get("summary", "")

    return {
        "memory_context": memory_context,
        "summary": summary
    }


# 🔬 路由 2：终审质检员（全面检查幻觉与切题度）
def grade_generation_route(state: GraphState) -> str:
    print("🔬 [质检路由] 正在严格审查生成的答案质量（防幻觉/防跑题）...")
    
    if state.get("retries", 0) >= 2:
        print("⚠️ [质检路由] 重试次数已达上限，不再打回，直接放行给用户。")
        return "end"

    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]
    
    # 【修改部分】：改用 JsonOutputParser
    parser = JsonOutputParser()
    
    prompt = f"""你是一个吹毛求疵的论文盲审专家。请根据【参考资料】，对大模型生成的【回答】进行严格质量检验。
    
评分标准：
1. 【回答】中的核心事实必须完全基于【参考资料】，不能有任何凭空捏造（防幻觉）。
2. 【回答】必须真正解决了用户的【问题】，不能答非所问。

满足以上两条输出 yes，否则输出 no。

【输出格式要求】：
必须输出一个标准的 JSON 字符串，包含一个 key 叫 "binary_score"，其值只能是 "yes" 或 "no"。
示例：{{"binary_score": "yes"}}

【参考资料】：{documents}
【用户问题】：{question}
【生成的回答】：{answer}"""
    
    response = llm.invoke(prompt)
    
    try:
        score_dict = parser.invoke(response)
        grade = score_dict.get("binary_score", "yes")
    except Exception as e:
        print(f"⚠️ [质检路由] 解析异常，默认通过。错误：{e}")
        grade = "yes"
        
    if str(grade).lower() == "yes":
        print("🎉 [质检路由] 完美通关！回答质量极高，允许输出给老板！")
        return "end"
    else:
        print("🚨 [质检路由] 发现幻觉或答非所问！拒绝接收，触发改写机制重搜！")
        return "rewrite"



# 5. 组装终极图网络
# ==========================================
print("🏗️ 正在组装 Agentic RAG 图网络...")
workflow = StateGraph(GraphState)

# 注册所有打工人节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("tool_router", tool_router)
workflow.add_node("generate_direct", generate_direct)
workflow.add_node("memory_node", memory_node)

# 路线起点
workflow.add_edge(START, "tool_router")
workflow.add_edge("memory_node", "generate_direct")
workflow.add_conditional_edges(
    "tool_router",
    lambda state: state["tool"],
    {
        "rag": "retrieve",
        "llm": "generate_direct",
        "memory": "memory_node"
    }
)


# 核心条件路由 1：检索完成后，过初审裁判
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_route,
    {
        "generate": "generate",
        "rewrite_question": "rewrite_question"
    }
)

# 核心条件路由 2：生成完成后，过终审质检员
workflow.add_conditional_edges(
    "generate",
    grade_generation_route,
    {
        "end": END,
        "rewrite": "rewrite_question"  # 质检失败，直接回炉改写关键词重搜
    }
)

# 改写完问题后，强行拐回检索员节点重新搜
workflow.add_edge("rewrite_question", "retrieve")

# 编译应用
app = workflow.compile()


# ==========================================
# 6. 点火发射！
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Agentic RAG (自我迭代反思特工) 启动！\n" + "="*50)
    
    # 🎯 测试提示：
    # 可以用一个在 PDF 里根本找不到的事实、或者需要深度泛化理解的问题来刁难它
    test_question = "2024年4月-2024年10月经历了什么"  
    
    print(f"🙋 老板：{test_question}\n")
    
    initial_state = {"question": test_question, "retries": 0, "history": ""}  # 本地测试必须补齐字段，否则报错
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print(f"🤖 最终安全回答：\n{final_state['answer']}")