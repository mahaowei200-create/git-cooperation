from memory.vector_memory import VectorMemory
from memory.sqlite_memory import get_history, get_summary

class MemoryOrchestrator:
    def __init__(self):
        self.vector_memory = VectorMemory()

    def retrieve(self, session_id: str, question: str):
        """
        🧠 统一 memory 调度入口
        """

        # 1️⃣ vector memory（最高优先级）
        vector_mem = self.vector_memory.search(question, top_k=3)

        # 2️⃣ summary memory
        summary = get_summary(session_id)

        # 3️⃣ short-term memory
        history = get_history(session_id)
        history_text = "\n".join([f"{r}:{c}" for r, c in history])

        return {
            "vector_memory": "\n".join(vector_mem), 
            "summary": summary,
            "history": history_text
        }

    def add(self, session_id: str, role: str, content: str):
        """
        🧠 写入 memory（统一入口）
        """
        self.vector_memory.add(f"{role}: {content}")