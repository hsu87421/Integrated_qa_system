# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    # @staticmethod
    # def rag_prompt():
    #     # 创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )
    #
    #     # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def rag_prompt():
        '''添加了历史记录，注意用在：new_rag_system'''
        return PromptTemplate(
            template="""
        你是企业员工服务助手，面向全体员工回答入职、考勤休假、财务审批、行政和 IT 服务问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 先判断上下文是否直接回答当前问题。仅当上下文提供了相关企业制度时，才按制度回答；不得用常识补充或编造公司规则。
           - 若上下文为空、无关或没有覆盖当前问题，忽略这些上下文，基于通用知识直接回答，并在开头注明“以下为通用信息，不代表公司内部制度”。
           - 企业制度上下文提供制度名称、版本或生效日期时，必须在答案中注明。
           - 回答企业办理事项时，优先按“适用范围、办理步骤、所需材料、审批节点、注意事项”输出。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确的回答，避免无关信息。
           - 不要因为没有企业制度上下文而拒绝回答通用问题；只有用户明确要求公司内部规则且上下文未覆盖时，才说明“当前知识库没有找到明确制度依据”。
        **对话历史**:
         {history}
        **上下文**:
         {context}
        **问题**:
         {question}

        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
               问题: {query}  
               假设答案:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询分解为多个简单子查询，每行一个子查询，最多生成两个子查询（只保留子查询问题，其他的文本都不需要）：
               eg: 
               用户原始query："新员工入职和转正分别需要办理哪些事项？"
               子查询："新员工入职需要办理哪些事项？"，"员工转正需要办理哪些事项？"
               
               查询: {query}  
               子查询:  
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
               将以下复杂查询简化为一个更简单的问题：  
               查询: {query}  
               简化问题:  
               """,
            #   定义输入变量
            input_variables=["query"]
        )
if __name__ == '__main__':
    # rga_prompt = RAGPrompts.rag_prompt()
    # result = rga_prompt.format(context="员工服务制度", question="调休如何申请", phone="企业服务台")
    # print(f'result-->{result}')
    hyde = RAGPrompts.subquery_prompt()
    result = hyde.format(query="调休和年假申请有什么区别")
    print(result)
