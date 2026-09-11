import os
import config.config as config
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from document_parser import load_and_chunk_pdf
from retriever import HybridRetriever

load_dotenv()

chunks = load_and_chunk_pdf(config.PDF_PATH)

engine = HybridRetriever(embed_model = config.EMBED_MODEL_NAME,rerank_model=config.RERANK_MODEL_NAME)

engine.ingest_docs(chunks)

def langchain_adapter_retriever(query:str)->str:
    result = engine.search(query,top_k=5,final_k=3)
    context = "\n\n".join([doc for doc,score in result])
    return context if context else "未检索到"

model = ChatOpenAI(model = "deepseek-chat")

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(
    """
    你是一个极其严谨的企业内部知识库 AI。请严格根据以下【参考资料】回答【问题】。
    如果资料中未包含答案，请直接回复“抱歉，内部资料未提及”。
    【参考资料】
    {context}
    【用户问题】
    {question}
    """
)

chain = (
    {"context":langchain_adapter_retriever,"question":RunnablePassthrough()}
    | prompt
    | model
    | parser
    )


while True:
    user_input = input("(老板):输入q离开:")
    if user_input.lower() == "q":
        break
    final_answer = chain.invoke(user_input)
    print(final_answer)
