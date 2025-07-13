from typing import Dict, Any, List, Optional
import json
import re
import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from config import Config
from .rag_service import RAGService
from .session_service import SessionService

class AIService:
    def __init__(self):
        """初始化AI服务"""
        print("🤖 初始化AI服务...")
        
        # 设置环境变量
        os.environ['OPENAI_API_KEY'] = Config.OPENAI_API_KEY
        
        # 初始化OpenAI客户端
        try:
            self.openai_client = OpenAI(
                api_key=Config.OPENAI_API_KEY,
                base_url=Config.OPENAI_BASE_URL
            )
            print("✅ OpenAI客户端初始化成功")
        except Exception as e:
            print(f"⚠️  OpenAI客户端初始化失败: {e}")
            self.openai_client = None
            
        # 初始化DeepSeek客户端
        try:
            self.deepseek_client = OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL
            )
            print("✅ DeepSeek客户端初始化成功")
        except Exception as e:
            print(f"⚠️  DeepSeek客户端初始化失败: {e}")
            self.deepseek_client = None
        
        # 初始化服务组件
        self.rag_service = RAGService()
        self.session_service = SessionService()
        
        print("✅ AI服务初始化完成")
    
    def _call_llm(self, messages: List[Dict[str, str]], use_deepseek: bool = False, 
                  temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """调用大模型"""
        try:
            if use_deepseek and self.deepseek_client:
                client = self.deepseek_client
                model = Config.DEEPSEEK_MODEL
                model_name = "DeepSeek"
            elif self.openai_client:
                client = self.openai_client
                model = Config.OPENAI_MODEL
                model_name = "OpenAI"
            else:
                return {
                    "success": False,
                    "error": "没有可用的AI客户端",
                    "model_used": "无"
                }
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model_used": model_name,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM调用失败: {str(e)}",
                "model_used": model_name if 'model_name' in locals() else "未知"
            }
    
    def _build_context_aware_messages(self, question: str, session_id: str = None, 
                                    relevant_docs: List[Dict] = None, document_ids: List[str] = None) -> List[Dict[str, str]]:
        """构建上下文感知的消息列表"""
        messages = []
        
        # 系统提示词
        system_prompt = """你是一个智能学习助手，擅长基于文档内容和对话历史提供准确、有用的回答。

核心能力：
1. **文档理解**：深度分析用户上传的文档，提取关键信息
2. **上下文记忆**：记住之前的对话内容，提供连贯的多轮对话体验
3. **智能推理**：结合文档内容和对话历史，进行深入分析和推理
4. **学习辅导**：帮助用户理解概念、生成总结、制定学习计划

回答规则：
- 如果有相关文档，优先基于文档内容回答，并明确引用来源
- 记住用户之前问过的问题，避免重复解释已知概念
- 如果用户问题与之前的对话相关，要体现出连续性
- 在没有文档支持时，诚实说明并提供一般性建议
- 保持回答的准确性、清晰性和有用性
- 使用中文回答，语言要专业但易懂

特别注意：
- 理解用户问题的上下文背景
- 适当引用之前的对话内容
- 提供具有连贯性的回答"""

        messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史对话上下文
        if session_id:
            historical_messages = self.session_service.get_context_for_llm(session_id)
            if historical_messages:
                # 在系统消息后添加历史对话
                messages.extend(historical_messages)
        
        # 构建当前问题的消息
        if relevant_docs:
            # 有相关文档的情况
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc["metadata"].get("source", f"文档{i}")
                content = doc["content"]
                context_parts.append(f"【文档{i}: {source}】\n{content}")
            
            context = "\n\n".join(context_parts)
            
            user_message = f"""请基于以下文档内容回答我的问题：

=== 相关文档内容 ===
{context}

=== 我的问题 ===
{question}

请根据上述文档内容回答问题。如果文档中有相关信息，请详细解释；如果没有直接相关的信息，请说明并提供一般性的建议。"""
        else:
            # 没有相关文档的情况
            if document_ids and len(document_ids) > 0:
                # 绑定了文档但没找到相关内容
                user_message = f"""我的问题是：{question}

注意：在绑定的 {len(document_ids)} 个文档中没有找到相关内容。请基于你的知识回答问题，或建议检查文档内容是否包含相关信息。"""
            else:
                # 用户主动选择不绑定文档
                user_message = f"""我的问题是：{question}

注意：当前处于纯大模型回答模式，没有绑定任何文档。我将基于我的训练知识为你回答问题。"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def answer_question(self, question: str, use_deepseek: bool = False, session_id: str = None, document_ids: List[str] = None) -> Dict[str, Any]:
        """智能问答（支持上下文和文档检索）"""
        try:
            # 如果没有提供session_id，生成一个
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 根据是否指定文档ID来决定是否使用文档
            if document_ids and len(document_ids) > 0:
                # 搜索指定文档的内容（支持多个文档）
                relevant_docs = self._search_in_multiple_documents(question, document_ids, k=5)
            else:
                # 不使用任何文档，纯大模型回答
                relevant_docs = []
            
            sources = []
            if relevant_docs:
                for doc in relevant_docs:
                    source_info = {
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "filename": doc["metadata"].get("source", ""),
                        "score": doc.get("score", 0)
                    }
                    sources.append(source_info)
            
            # 构建上下文感知的消息
            messages = self._build_context_aware_messages(question, session_id, relevant_docs, document_ids)
            
            # 调用LLM
            result = self._call_llm(messages, use_deepseek, temperature=0.3)
            
            if not result["success"]:
                return result
            
            # 保存对话到会话历史
            self.session_service.add_conversation(
                session_id=session_id,
                question=question,
                answer=result["content"],
                model_used=result["model_used"],
                sources=sources
            )
            
            return {
                "success": True,
                "answer": result["content"],
                "sources": sources,
                "model_used": result["model_used"],
                "has_context": len(relevant_docs) > 0,
                "session_id": session_id,
                "tokens_used": result.get("tokens_used", 0),
                "context_used": len(self.session_service.get_conversation_history(session_id)) > 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"问答处理失败: {str(e)}",
                "session_id": session_id
            }
    
    def generate_summary(self, documents: List[Dict[str, Any]], use_deepseek: bool = False) -> Dict[str, Any]:
        """生成文档总结"""
        try:
            if not documents:
                return {
                    "success": False,
                    "error": "没有提供文档"
                }
            
            # 合并文档内容
            content_parts = []
            sources = []
            
            for doc in documents:
                content_parts.append(doc["content"])
                sources.append({
                    "filename": doc["metadata"].get("source", ""),
                    "content_preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
                })
            
            combined_content = "\n\n".join(content_parts)
            
            # 构建总结消息
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的文档总结专家。请为用户生成高质量的文档总结。

总结要求：
1. **结构清晰**：使用明确的标题和层次结构
2. **内容完整**：涵盖文档的主要观点和关键信息
3. **逻辑合理**：按照逻辑顺序组织内容
4. **简洁明了**：去除冗余，突出要点
5. **易于理解**：使用清晰的语言表达

总结格式：
- 使用中文撰写
- 采用markdown格式
- 包含主要章节和要点
- 适当使用列表和强调"""
                },
                {
                    "role": "user",
                    "content": f"""请为以下文档内容生成详细的总结：

{combined_content}

请生成一个结构化的总结，包括：
1. 文档概述
2. 主要内容要点
3. 关键信息提取
4. 总结性观点"""
                }
            ]
            
            # 调用LLM生成总结
            result = self._call_llm(messages, use_deepseek, temperature=0.3, max_tokens=3000)
            
            if not result["success"]:
                return result
            
            return {
                "success": True,
                "summary": result["content"],
                "sources": sources,
                "model_used": result["model_used"],
                "document_count": len(documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"总结生成失败: {str(e)}"
            }
    
    def generate_quiz(self, documents: List[Dict[str, Any]], question_count: int = 5, 
                     use_deepseek: bool = False) -> Dict[str, Any]:
        """生成练习题"""
        try:
            if not documents:
                return {
                    "success": False,
                    "error": "没有提供文档"
                }
            
            # 合并文档内容
            content_parts = []
            for doc in documents:
                content_parts.append(doc["content"])
            
            combined_content = "\n\n".join(content_parts)
            
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的教育评估专家，擅长根据学习材料生成高质量的练习题。

题目要求：
1. **难度适中**：既要有一定挑战性，又要确保可以解答
2. **类型多样**：包含选择题、判断题、简答题等
3. **覆盖全面**：涵盖文档的主要知识点
4. **表述清晰**：题目和选项表述准确、无歧义
5. **答案准确**：提供正确答案和详细解释

返回格式：
- 使用严格的JSON格式
- 每道题包含：题目、类型、选项（如适用）、正确答案、解释
- 确保JSON格式正确，可以被解析"""
                },
                {
                    "role": "user",
                    "content": f"""基于以下文档内容，生成 {question_count} 道练习题：

{combined_content}

请生成JSON格式的练习题，格式如下：
{{
    "questions": [
        {{
            "id": 1,
            "type": "choice",
            "question": "题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "correct_answer": "A",
            "explanation": "答案解释"
        }},
        {{
            "id": 2,
            "type": "judge",
            "question": "判断题内容",
            "correct_answer": "true",
            "explanation": "答案解释"
        }},
        {{
            "id": 3,
            "type": "short_answer",
            "question": "简答题内容",
            "correct_answer": "参考答案",
            "explanation": "答案要点"
        }}
    ]
}}

注意：请确保返回有效的JSON格式，题目类型包括choice（选择题）、judge（判断题）、short_answer（简答题）。"""
                }
            ]
            
            result = self._call_llm(messages, use_deepseek, temperature=0.5, max_tokens=3000)
            
            if not result["success"]:
                return result
            
            # 尝试解析JSON
            try:
                quiz_data = json.loads(result["content"])
                if "questions" not in quiz_data:
                    raise ValueError("返回数据格式错误")
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取JSON部分
                content = result["content"]
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    quiz_data = json.loads(json_match.group())
                else:
                    return {
                        "success": False,
                        "error": "无法解析生成的练习题格式"
                    }
            
            return {
                "success": True,
                "quiz": quiz_data,
                "model_used": result["model_used"],
                "question_count": len(quiz_data.get("questions", []))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"练习题生成失败: {str(e)}"
            }
    
    def generate_study_plan(self, documents: List[Dict[str, Any]], difficulty: str = "medium", 
                           use_deepseek: bool = False) -> Dict[str, Any]:
        """生成学习计划"""
        try:
            if not documents:
                return {
                    "success": False,
                    "error": "没有提供文档"
                }
            
            # 合并文档内容
            content_parts = []
            for doc in documents:
                content_parts.append(doc["content"])
            
            combined_content = "\n\n".join(content_parts)
            
            difficulty_map = {
                "easy": "初级水平，适合入门学习",
                "medium": "中级水平，有一定基础",
                "hard": "高级水平，需要深入掌握"
            }
            
            difficulty_desc = difficulty_map.get(difficulty, "中级水平")
            
            messages = [
                {
                    "role": "system",
                    "content": f"""你是一个专业的学习规划师，擅长为学习者制定个性化的学习计划。

学习计划要求：
1. **目标明确**：设定清晰的学习目标和里程碑
2. **结构合理**：按照学习进度合理安排内容
3. **时间规划**：提供具体的时间安排建议
4. **方法指导**：包含学习方法和技巧建议
5. **可操作性**：计划具体可执行

当前设定难度：{difficulty_desc}

请使用中文，采用markdown格式，结构化地组织学习计划。"""
                },
                {
                    "role": "user",
                    "content": f"""基于以下学习材料，为{difficulty_desc}的学习者制定详细的学习计划：

{combined_content}

请生成包含以下部分的学习计划：
1. 学习目标
2. 内容概览
3. 学习阶段划分
4. 具体学习安排
5. 学习方法建议
6. 评估方式
7. 注意事项"""
                }
            ]
            
            result = self._call_llm(messages, use_deepseek, temperature=0.4, max_tokens=3000)
            
            if not result["success"]:
                return result
            
            return {
                "success": True,
                "study_plan": result["content"],
                "difficulty": difficulty,
                "model_used": result["model_used"],
                "document_count": len(documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"学习计划生成失败: {str(e)}"
            }
    
    def explain_concept(self, concept: str, use_deepseek: bool = False, session_id: str = None) -> Dict[str, Any]:
        """概念解释（支持上下文）"""
        try:
            # 如果没有提供session_id，生成一个
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # 搜索相关文档
            relevant_docs = self.rag_service.search_similar_documents(concept, k=3)
            
            # 构建解释消息
            messages = [
                {
                    "role": "system",
                    "content": """你是一个概念解释专家，擅长用清晰、易懂的方式解释各种概念。

解释要求：
1. **定义准确**：给出概念的准确定义
2. **通俗易懂**：用简单的语言解释复杂概念
3. **举例说明**：提供具体例子帮助理解
4. **关联分析**：说明与其他概念的关系
5. **实际应用**：介绍概念的实际应用场景

如果之前的对话中提到过相关概念，要体现出连续性和关联性。
使用中文回答，结构清晰，便于理解。"""
                }
            ]
            
            # 添加历史对话上下文
            if session_id:
                historical_messages = self.session_service.get_context_for_llm(session_id)
                if historical_messages:
                    messages.extend(historical_messages[-4:])  # 只取最近2轮对话
            
            # 构建当前问题
            if relevant_docs:
                context_parts = []
                for doc in relevant_docs:
                    context_parts.append(doc["content"])
                context = "\n\n".join(context_parts)
                
                user_message = f"""请解释以下概念：{concept}

相关文档内容：
{context}

请基于文档内容和你的知识，详细解释这个概念。"""
            else:
                user_message = f"""请详细解释以下概念：{concept}

请提供：
1. 概念定义
2. 关键特征
3. 具体例子
4. 实际应用
5. 相关概念"""
            
            messages.append({"role": "user", "content": user_message})
            
            # 调用LLM
            result = self._call_llm(messages, use_deepseek, temperature=0.3)
            
            if not result["success"]:
                return result
            
            # 保存对话到会话历史
            self.session_service.add_conversation(
                session_id=session_id,
                question=f"解释概念：{concept}",
                answer=result["content"],
                model_used=result["model_used"]
            )
            
            return {
                "success": True,
                "explanation": result["content"],
                "concept": concept,
                "model_used": result["model_used"],
                "has_context": len(relevant_docs) > 0,
                "session_id": session_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"概念解释失败: {str(e)}",
                "session_id": session_id
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        return self.session_service.get_session_info(session_id)
    
    def clear_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """清除会话上下文"""
        try:
            success = self.session_service.clear_session(session_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"会话 {session_id} 的上下文已清除"
                }
            else:
                return {
                    "success": False,
                    "error": "清除会话上下文失败"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"清除会话上下文失败: {str(e)}"
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        try:
            session_status = self.session_service.get_health_status()
            
            return {
                "ai_service": {
                    "openai_available": self.openai_client is not None,
                    "deepseek_available": self.deepseek_client is not None
                },
                "session_service": session_status,
                "rag_service": {
                    "document_count": len(self.rag_service.list_documents())
                }
            }
            
        except Exception as e:
            return {
                "error": f"获取服务状态失败: {str(e)}"
            }
    
    def _search_in_multiple_documents(self, question: str, document_ids: List[str], k: int = 5) -> List[Dict]:
        """在多个指定文档中搜索相关内容"""
        try:
            all_relevant_docs = []
            
            for document_id in document_ids:
                # 搜索每个文档
                docs_in_document = self._search_in_specific_document(question, document_id, k)
                all_relevant_docs.extend(docs_in_document)
            
            # 按相关性排序，取前k个
            all_relevant_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            return all_relevant_docs[:k]
            
        except Exception as e:
            print(f"多文档搜索错误: {str(e)}")
            return []
    
    def _search_in_specific_document(self, question: str, document_id: str, k: int = 5) -> List[Dict]:
        """在指定文档中搜索相关内容"""
        try:
            # 获取指定文档的所有内容
            document_contents = self.rag_service.get_documents_by_id(document_id)
            
            if not document_contents:
                return []
            
            # 简单的关键词匹配（你也可以使用更复杂的相似性搜索）
            relevant_docs = []
            for doc in document_contents:
                # 计算简单的相关性得分
                content = doc["content"].lower()
                question_lower = question.lower()
                
                # 基于关键词出现次数计算分数
                score = 0
                for word in question_lower.split():
                    if len(word) > 1:  # 忽略太短的词
                        score += content.count(word)
                
                if score > 0:
                    relevant_docs.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            
            # 按分数排序并返回前k个
            relevant_docs.sort(key=lambda x: x["score"], reverse=True)
            return relevant_docs[:k]
            
        except Exception as e:
            print(f"单文档搜索错误: {str(e)}")
            return [] 