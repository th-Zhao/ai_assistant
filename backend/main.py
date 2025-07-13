import uuid
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

from config import Config
from services.document_service import DocumentService
from services.rag_service import RAGService
from services.ai_service import AIService

# 创建FastAPI应用
app = FastAPI(
    title="智能学习助手",
    description="基于大模型API的智能学习助手系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
document_service = DocumentService()
rag_service = RAGService()
ai_service = AIService()

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    use_deepseek: bool = False
    session_id: str = None
    document_ids: List[str] = []  # 支持多个文档ID的列表

class QuizRequest(BaseModel):
    question_count: int = 5
    use_deepseek: bool = False

class StudyPlanRequest(BaseModel):
    difficulty: str = "medium"  # easy, medium, hard
    use_deepseek: bool = False

class ConceptRequest(BaseModel):
    concept: str
    use_deepseek: bool = False
    session_id: str = None

# API端点

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "智能学习助手 API - 增强版",
        "version": "2.0.0",
        "features": [
            "🤖 智能问答（支持上下文记忆）",
            "📚 文档上传与处理",
            "📝 自动总结生成",
            "❓ 练习题生成",
            "📋 学习计划制定",
            "💡 概念解释",
            "🗣️ 多轮对话支持",
            "🗃️  Redis会话管理",
            "🔧 服务状态监控"
        ],
        "endpoints": {
            # 核心功能
            "/docs": "📖 API文档",
            "/upload": "📤 上传文档",
            "/ask": "💬 智能问答（支持会话上下文）",
            "/summary": "📋 生成总结",
            "/quiz": "❓ 生成练习题",
            "/study-plan": "📅 生成学习计划",
            "/explain": "💡 概念解释（支持会话上下文）",
            
            # 文档管理
            "/documents": "📚 文档管理",
            "/documents/{id}": "🗑️ 删除文档",
            
            # 会话管理
            "/session/{session_id}": "🔍 获取会话信息",
            "/sessions": "📋 列出所有会话",
            "/conversation/{session_id}": "🧹 清除会话上下文",
            "/session/{session_id}/cleanup": "🗑️ 清理指定会话",
            "/sessions/cleanup": "🧹 清理过期会话",
            
            # 系统监控
            "/health": "❤️ 健康检查",
            "/service-status": "📊 详细服务状态"
        },
        "session_support": {
            "redis_storage": "使用Redis进行会话持久化",
            "context_memory": "支持多轮对话上下文",
            "auto_expire": "24小时自动过期",
            "max_history": "每会话最多20轮对话"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传并处理文档"""
    try:
        # 检查文件类型
        if not file.filename:
            raise HTTPException(status_code=400, detail="未选择文件")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型。支持的类型：{', '.join(Config.ALLOWED_EXTENSIONS)}"
            )
        
        # 检查文件大小
        content = await file.read()
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="文件大小超过限制（50MB）")
        
        # 处理文档
        result = document_service.process_document(content, file.filename)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 添加到向量存储
        document_id = str(uuid.uuid4())
        rag_result = rag_service.add_documents(result["documents"], document_id)
        
        if not rag_result["success"]:
            raise HTTPException(status_code=500, detail=f"向量存储失败：{rag_result['error']}")
        
        return {
            "success": True,
            "message": "文档上传成功",
            "document_id": document_id,
            "filename": result["filename"],
            "preview": result["preview"],
            "stats": {
                "text_length": result["text_length"],
                "chunk_count": result["chunk_count"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误：{str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """智能问答"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        result = ai_service.answer_question(request.question, request.use_deepseek, request.session_id, request.document_ids)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "model_used": result["model_used"],
            "session_id": result["session_id"],
            "has_context": result.get("has_context", False),
            "context_used": result.get("context_used", False),
            "tokens_used": result.get("tokens_used", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答服务错误：{str(e)}")

@app.post("/summary")
async def generate_summary(use_deepseek: bool = False):
    """生成文档总结"""
    try:
        # 获取所有文档
        documents = rag_service.list_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="没有可用的文档，请先上传文档")
        
        # 获取文档内容用于总结
        all_docs = []
        for doc_info in documents:
            doc_id = doc_info["document_id"]
            # 直接获取文档内容
            doc_contents = rag_service.get_documents_by_id(doc_id)
            all_docs.extend(doc_contents)
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="无法获取文档内容")
        
        result = ai_service.generate_summary(all_docs, use_deepseek)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "summary": result["summary"],
            "model_used": result["model_used"],
            "document_count": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"总结生成错误：{str(e)}")

@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """生成练习题"""
    try:
        if request.question_count < 1 or request.question_count > 20:
            raise HTTPException(status_code=400, detail="题目数量应在1-20之间")
        
        # 获取文档内容
        documents = rag_service.list_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="没有可用的文档，请先上传文档")
        
        # 获取文档内容用于生成练习题
        all_docs = []
        for doc_info in documents:
            doc_id = doc_info["document_id"]
            # 直接获取文档内容，而不是搜索
            doc_contents = rag_service.get_documents_by_id(doc_id)
            all_docs.extend(doc_contents)
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="无法获取文档内容")
        
        result = ai_service.generate_quiz(all_docs, request.question_count, request.use_deepseek)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "quiz": result["quiz"],
            "model_used": result["model_used"],
            "question_count": request.question_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"练习题生成错误：{str(e)}")

@app.post("/study-plan")
async def generate_study_plan(request: StudyPlanRequest):
    """生成学习计划"""
    try:
        if request.difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(status_code=400, detail="难度级别应为：easy, medium, hard")
        
        # 获取文档内容
        documents = rag_service.list_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="没有可用的文档，请先上传文档")
        
        # 获取文档内容用于生成学习计划
        all_docs = []
        for doc_info in documents:
            doc_id = doc_info["document_id"]
            # 直接获取文档内容
            doc_contents = rag_service.get_documents_by_id(doc_id)
            all_docs.extend(doc_contents)
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="无法获取文档内容")
        
        result = ai_service.generate_study_plan(all_docs, request.difficulty, request.use_deepseek)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "study_plan": result["study_plan"],
            "difficulty": result["difficulty"],
            "model_used": result["model_used"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"学习计划生成错误：{str(e)}")

@app.post("/explain")
async def explain_concept(request: ConceptRequest):
    """概念解释"""
    try:
        if not request.concept.strip():
            raise HTTPException(status_code=400, detail="概念不能为空")
        
        result = ai_service.explain_concept(request.concept, request.use_deepseek, request.session_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "concept": result["concept"],
            "explanation": result["explanation"],
            "has_context": result["has_context"],
            "model_used": result["model_used"],
            "session_id": result["session_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"概念解释错误：{str(e)}")

@app.get("/documents")
async def list_documents():
    """获取文档列表"""
    try:
        documents = rag_service.list_documents()
        stats = rag_service.get_vector_store_stats()
        
        return {
            "success": True,
            "documents": documents,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表错误：{str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """删除指定文档"""
    try:
        result = rag_service.delete_document(document_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档错误：{str(e)}")

@app.delete("/documents")
async def clear_all_documents():
    """清空所有文档"""
    try:
        result = rag_service.clear_all_documents()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空文档错误：{str(e)}")

@app.delete("/conversation/{session_id}")
async def clear_conversation_context(session_id: str):
    """清除会话上下文"""
    try:
        result = ai_service.clear_conversation_context(session_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除会话上下文错误：{str(e)}")

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息和历史"""
    try:
        session_info = ai_service.get_session_info(session_id)
        
        return {
            "success": True,
            "session_info": session_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话信息错误：{str(e)}")

@app.get("/sessions")
async def list_sessions():
    """获取所有活跃会话"""
    try:
        sessions = ai_service.session_service.list_active_sessions()
        
        # 获取每个会话的基本信息
        session_list = []
        for session_id in sessions:
            session_info = ai_service.session_service.get_session_info(session_id)
            # 只返回基本信息，不包括完整的对话历史
            session_summary = {
                "session_id": session_id,
                "conversation_count": session_info.get("conversation_count", 0),
                "last_activity": session_info.get("last_activity", 0),
                "created_at": session_info.get("created_at", 0)
            }
            session_list.append(session_summary)
        
        return {
            "success": True,
            "sessions": session_list,
            "total_sessions": len(session_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表错误：{str(e)}")

@app.get("/service-status")
async def get_service_status():
    """获取详细的服务状态"""
    try:
        status = ai_service.get_service_status()
        
        return {
            "success": True,
            "service_status": status,
            "timestamp": int(__import__('time').time())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取服务状态错误：{str(e)}")

@app.post("/session/{session_id}/cleanup")
async def cleanup_session(session_id: str):
    """清理指定会话的过期数据"""
    try:
        # 清理会话数据
        success = ai_service.session_service.clear_session(session_id)
        
        if success:
            return {
                "success": True,
                "message": f"会话 {session_id} 已清理"
            }
        else:
            raise HTTPException(status_code=404, detail="会话不存在或清理失败")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理会话错误：{str(e)}")

@app.post("/sessions/cleanup")
async def cleanup_expired_sessions():
    """清理所有过期的会话"""
    try:
        cleaned_count = ai_service.session_service.cleanup_expired_sessions()
        
        return {
            "success": True,
            "message": f"已清理 {cleaned_count} 个过期会话",
            "cleaned_sessions": cleaned_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理过期会话错误：{str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        stats = rag_service.get_vector_store_stats()
        return {
            "status": "healthy",
            "service": "智能学习助手",
            "version": "1.0.0",
            "documents": stats.get("total_documents", 0),
            "chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    ) 