from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

# 直接导入已经编译好的graph，禁止再次compile
from app.agents.personal_chief import graph

app = FastAPI(title="LangGraph Local Server")


@app.post("/invoke")
async def invoke(payload: dict):
    """一次性调用，拿到完整返回结果"""
    result = await graph.ainvoke(payload.get("input", {}))
    return {"result": result}


@app.post("/stream")
async def stream(payload: dict):
    """流式输出，对应 astream"""
    async def event_generator():
        async for chunk in graph.astream(payload.get("input", {})):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=2024, reload=True)
