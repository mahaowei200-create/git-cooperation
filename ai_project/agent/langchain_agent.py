from chromadb import config
import config.config as config
from langchain.agents import create_agent
from langchain.tools  import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rag.document_parser import load_and_chunk_pdf
from rag.retriever import HybridRetriever

load_dotenv()

llm = ChatOpenAI(model = "deepseek-chat",temperature = 0.1)
chunks = load_and_chunk_pdf(config.PDF_PATH)
engine = HybridRetriever(embed_model=config.EMBED_MODEL_NAME,rerank_model=config.RERANK_MODEL_NAME)
engine.ingest_docs(chunks)

@tool
def research_weather(city:str)->str:
    """当需要查询天气的时候调用这个函数"""
    mock_data = {"北京": "雷阵雨，建议携带雨伞","上海": "晴，35 摄氏度","武汉": "大雾，能见度较低",}
    return mock_data.get(city,"未查询到天气")
@tool
def calculate_entire_money(base: int, bonus: int) -> str:
    """ 这是一个计算总工资的方法，当需要计算总工资的时候，调用这个函数  """
    print(f"\n[🧮 工具触发] 正在计算总工资……")
    total = base + bonus
    return f"总工资为{total}"
@tool
def search_internal_knowledge(query: str) -> str:
    """查询个人简历经历、项目经验、个人信息等 PDF 资料时，必须调用此工具。"""
    docs = engine.search(query,top_k = 5,final_k = 3)
    answer = "\n\n".join([doc for doc,score in docs])
    return answer if answer else "未检索到相关信息"

tools = [research_weather,calculate_entire_money,search_internal_knowledge]

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt=(
        "你是一个非常专业的智能助手，去解决相应的问题"
        "当用户询问的问题适合用相关的工具时，一定使用相关工具"
        "获得工具结束后 ，用简洁的中文语言清晰的回答用户"
    )
)

def ask_agent(query:str)->str:
    result = agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":query
                }
            ]
        }
    )
    final_message = result["messages"][-1]
    print(f"答案：{final_message.content}")


if __name__ == "__main__":
    ask_agent("2024年4月到2024年十月经历了什么")
