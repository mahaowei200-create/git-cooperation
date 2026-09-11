from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

def summarize_history(history_text: str, old_memory: str = "") -> str:
    """
    长期记忆 = 旧记忆 + 新对话
    """

    prompt = f"""
你是一个记忆压缩器，请更新长期记忆。

【旧记忆】
{old_memory}
 
【新对话】
{history_text}

要求：
1. 合并信息
2. 去重
3. 更新用户状态
4. 保留重要事实（身份/偏好/项目）

输出最终记忆（简洁中文）：
"""

    response = llm.invoke(prompt)
    return response.content