from fastapi import FastAPI
from core.agentic_rag import app as rag_app
from memory.sqlite_memory import init_db,save_message,get_history,save_summary,get_summary
from memory.summarizer import summarize_history
from pydantic import BaseModel
from memory.vector_memory import VectorMemory
from memory.memory_orchestrator import MemoryOrchestrator


memory = MemoryOrchestrator()
app = FastAPI()
init_db()

class ChatRequest(BaseModel):
    question:str
    session_id:str="default"

@app.get("/")
def home():
    return {"message":"FastAPI + RAG 已启动 🚀"}

@app.post("/chat")
def chat(req:ChatRequest):
    session_id = req.session_id
    user_question = req.question

    # 1. 先读取现有历史、旧摘要（只查一次数据库）
    history = get_history(session_id)
    old_summary = get_summary(session_id)
    history_text = "\n".join([f"{role}:{content}" for role,content in history])

    print("📌 HISTORY:", history)
    print("📌 OLD SUMMARY:", old_summary)

    # 2. 先用旧摘要调用RAG生成回答
   # 🧠 召回语义记忆（核心）
    mem_ctx = memory.retrieve(session_id, user_question)

    memory_text = mem_ctx["vector_memory"]
    summary_text = mem_ctx["summary"]
    history_text = mem_ctx["history"]

    result = rag_app.invoke({
    "history": history_text,
    "summary": summary_text,
    "memory": memory_text,
    "question": user_question,
    "retries": 0
})
    answer = result.get("answer","")

    # 3. 问答闭环：一次性保存用户提问 + AI回答
    save_message(session_id,"user", user_question)
    memory.add(session_id, "user", user_question)
    save_message(session_id,"assistant", answer)
    memory.add(session_id, "assistant", answer)
    print("✅ 已写入本轮完整 user + assistant 对话")

    # 4. 读取更新后的完整历史，判断是否更新长期记忆
    full_new_history = get_history(session_id)
    new_history_text = "\n".join([f"{r}:{c}" for r,c in full_new_history])
    use_summary = old_summary

    # 累计对话条数≥4 执行增量压缩更新记忆
    if len(full_new_history) >= 4:
        use_summary = summarize_history(new_history_text, old_summary)
        # 【核心修复】生成新摘要必须入库保存
        save_summary(session_id, use_summary)
        print("🧠 长期记忆已合并更新并持久化")

    return {
        "question": user_question,
        "session_id": session_id,
        "answer": answer
    }