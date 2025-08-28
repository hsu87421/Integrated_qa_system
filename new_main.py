"""Application service layer for enterprise policy RAG queries."""
from collections import defaultdict
import time
import uuid

from openai import OpenAI

from base import Config, logger
from rag_qa import RAGSystem, VectorStore


class IntegratedQASystem:
    """Coordinates streaming answers and short-lived in-memory conversation history."""

    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.client = OpenAI(
            api_key=self.config.DASHSCOPE_API_KEY,
            base_url=self.config.DASHSCOPE_BASE_URL,
        )
        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        self._session_histories = defaultdict(list)

    def call_dashscope(self, prompt):
        """Call the configured OpenAI-compatible model with streaming enabled."""
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业、谨慎的企业员工服务助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as error:
            self.logger.error("LLM 调用失败: %s", error)
            yield "抱歉，当前无法生成回答，请稍后重试。"

    def get_session_history(self, session_id):
        return list(self._session_histories.get(session_id, []))

    def update_session_history(self, session_id, question, answer):
        history = self._session_histories[session_id]
        history.append({"question": question, "answer": answer})
        del history[:-5]
        return list(history)

    def clear_session_history(self, session_id):
        self._session_histories.pop(session_id, None)
        return True

    def query(self, query, source_filter=None, session_id=None):
        """Yield a streaming answer and persist its completed text in session history."""
        started_at = time.time()
        history = self.get_session_history(session_id) if session_id else []
        answer = ""
        for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
            answer += token
            yield token, False

        if session_id:
            self.update_session_history(session_id, query, answer)
        self.logger.info("查询完成，耗时 %.2f 秒", time.time() - started_at)
        yield "", True


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("欢迎使用企业员工服务与制度智能问答系统。输入 exit 退出。")
    while True:
        query = input("\n输入问题: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        print("\n回答: ", end="", flush=True)
        for token, is_complete in qa_system.query(query, session_id=session_id):
            print(token, end="", flush=True)
            if is_complete:
                print()


if __name__ == "__main__":
    main()
