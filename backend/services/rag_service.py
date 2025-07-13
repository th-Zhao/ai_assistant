import os
import uuid
import json
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.schema import Document

from config import Config

class RAGService:
    def __init__(self):
        """初始化RAG服务"""
        # 创建向量存储目录
        os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)
        
        # 持久化文件路径
        self.documents_file = os.path.join(Config.VECTOR_STORE_PATH, "documents.pkl")
        self.metadata_file = os.path.join(Config.VECTOR_STORE_PATH, "metadata.json")
        
        # 初始化存储
        self.documents = {}  # {document_id: [Document]}
        self.document_metadata = {}
        
        # 加载已存储的数据
        self._load_documents()
        
        print(f"📚 RAG服务已启动，已加载 {len(self.documents)} 个文档")
    
    def _save_documents(self):
        """保存文档到文件"""
        try:
            # 保存文档内容（使用pickle保存Document对象）
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # 保存元数据（使用JSON）
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
                
            print(f"💾 已保存 {len(self.documents)} 个文档到磁盘")
            
        except Exception as e:
            print(f"❌ 保存文档失败: {str(e)}")
    
    def _load_documents(self):
        """从文件加载文档"""
        try:
            # 加载文档内容
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"📖 从磁盘加载了 {len(self.documents)} 个文档")
            
            # 加载元数据
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                print(f"📊 加载了 {len(self.document_metadata)} 条文档元数据")
                
        except Exception as e:
            print(f"⚠️  加载文档失败，使用空存储: {str(e)}")
            self.documents = {}
            self.document_metadata = {}
    
    def add_documents(self, documents: List[Document], document_id: str) -> Dict[str, Any]:
        """添加文档到存储"""
        try:
            if not documents:
                return {"success": False, "error": "没有文档可添加"}
            
            # 为每个文档添加唯一ID和文档ID
            for i, doc in enumerate(documents):
                doc.metadata["doc_id"] = f"{document_id}_{i}"
                doc.metadata["document_id"] = document_id
                doc.metadata["created_at"] = datetime.now().isoformat()
            
            # 添加到内存存储
            self.documents[document_id] = documents
            
            # 保存文档元数据
            self.document_metadata[document_id] = {
                "document_id": document_id,
                "filename": documents[0].metadata.get("source", ""),
                "chunk_count": len(documents),
                "created_at": datetime.now().isoformat(),
                "total_chars": sum(len(doc.page_content) for doc in documents)
            }
            
            # 立即保存到文件
            self._save_documents()
            
            return {
                "success": True,
                "message": f"成功添加 {len(documents)} 个文档块",
                "document_id": document_id,
                "chunk_count": len(documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"添加文档失败: {str(e)}"
            }
    
    def search_similar_documents(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """搜索相似文档（简化版本：基于关键词匹配）"""
        try:
            if not query.strip():
                return []
            
            query_lower = query.lower()
            results = []
            
            # 遍历所有文档进行简单的文本匹配
            for doc_id, docs in self.documents.items():
                for doc in docs:
                    content_lower = doc.page_content.lower()
                    
                    # 简单的相关性计算：查询词在文档中的出现次数
                    query_words = query_lower.split()
                    matches = sum(1 for word in query_words if word in content_lower)
                    score = matches / len(query_words) if query_words else 0
                    
                    if score >= score_threshold:
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": score
                        })
            
            # 按相关性排序并返回前k个结果
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"搜索文档时出错: {str(e)}")
            return []
    
    def get_documents_by_id(self, document_id: str) -> List[Dict[str, Any]]:
        """根据文档ID获取所有相关文档块"""
        try:
            if document_id not in self.documents:
                return []
            
            documents = []
            for doc in self.documents[document_id]:
                documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return documents
            
        except Exception as e:
            print(f"获取文档时出错: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """删除指定文档"""
        try:
            if document_id not in self.documents:
                return {"success": False, "error": "文档不存在"}
            
            chunk_count = len(self.documents[document_id])
            
            # 删除文档
            del self.documents[document_id]
            
            # 删除元数据
            if document_id in self.document_metadata:
                del self.document_metadata[document_id]
            
            # 保存更改到文件
            self._save_documents()
            
            return {
                "success": True,
                "message": f"成功删除文档 {document_id}",
                "deleted_chunks": chunk_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"删除文档失败: {str(e)}"
            }
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """清空所有文档"""
        try:
            total_chunks = sum(len(docs) for docs in self.documents.values())
            
            # 清空文档
            self.documents.clear()
            
            # 清空元数据
            self.document_metadata.clear()
            
            # 保存更改到文件
            self._save_documents()
            
            return {
                "success": True,
                "message": "已清空所有文档",
                "deleted_chunks": total_chunks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"清空文档失败: {str(e)}"
            }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        try:
            return list(self.document_metadata.values())
            
        except Exception as e:
            print(f"列出文档时出错: {str(e)}")
            return []
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            total_chunks = sum(len(docs) for docs in self.documents.values())
            total_documents = len(self.documents)
            
            total_chars = sum(
                sum(len(doc.page_content) for doc in docs)
                for docs in self.documents.values()
            )
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "average_chunk_size": total_chars // total_chunks if total_chunks > 0 else 0,
                "vector_store_path": Config.VECTOR_STORE_PATH
            }
            
        except Exception as e:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "error": str(e)
            }
    
    def get_relevant_context(self, query: str, max_chars: int = 4000) -> str:
        """获取与查询相关的上下文"""
        try:
            # 搜索相关文档
            results = self.search_similar_documents(query, k=10)
            
            if not results:
                return ""
            
            # 按相关性排序并组合上下文
            context_parts = []
            total_chars = 0
            
            for result in results:
                content = result["content"]
                if total_chars + len(content) > max_chars:
                    # 如果加上这段内容会超过限制，截取部分内容
                    remaining_chars = max_chars - total_chars
                    if remaining_chars > 100:  # 确保有足够的字符
                        content = content[:remaining_chars] + "..."
                    else:
                        break
                
                context_parts.append(content)
                total_chars += len(content)
                
                if total_chars >= max_chars:
                    break
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"获取上下文时出错: {str(e)}")
            return "" 