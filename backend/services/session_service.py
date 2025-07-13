import redis
import json
import time
from typing import Dict, List, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

class SessionService:
    def __init__(self):
        """初始化会话服务"""
        print("🗃️  初始化会话服务...")
        
        self.redis_client = None
        self.fallback_storage = {}  # 备用内存存储
        
        try:
            # 尝试连接Redis
            self.redis_client = redis.Redis(**Config.get_redis_config())
            # 测试连接
            self.redis_client.ping()
            print("✅ Redis连接成功")
        except Exception as e:
            print(f"⚠️  Redis连接失败，使用内存存储: {e}")
            self.redis_client = None
        
        print("✅ 会话服务初始化完成")
    
    def _get_session_key(self, session_id: str) -> str:
        """获取会话在Redis中的键名"""
        return f"session:{session_id}"
    
    def _get_context_key(self, session_id: str) -> str:
        """获取会话上下文在Redis中的键名"""
        return f"context:{session_id}"
    
    def add_conversation(self, session_id: str, question: str, answer: str, 
                        model_used: str = "", sources: List[Dict] = None) -> bool:
        """添加对话到会话历史"""
        try:
            conversation_item = {
                "timestamp": int(time.time()),
                "question": question,
                "answer": answer,
                "model_used": model_used,
                "sources": sources or []
            }
            
            if self.redis_client:
                # 使用Redis存储
                context_key = self._get_context_key(session_id)
                
                # 获取现有对话历史
                conversations = self.get_conversation_history(session_id)
                conversations.append(conversation_item)
                
                # 限制历史记录长度
                if len(conversations) > Config.MAX_CONVERSATION_HISTORY:
                    conversations = conversations[-Config.MAX_CONVERSATION_HISTORY:]
                
                # 存储到Redis
                self.redis_client.setex(
                    context_key, 
                    Config.SESSION_EXPIRE_TIME, 
                    json.dumps(conversations, ensure_ascii=False)
                )
                
                # 更新会话元数据
                session_key = self._get_session_key(session_id)
                session_meta = {
                    "last_activity": int(time.time()),
                    "conversation_count": len(conversations),
                    "created_at": int(time.time())
                }
                
                existing_meta = self.redis_client.get(session_key)
                if existing_meta:
                    existing_meta = json.loads(existing_meta)
                    session_meta["created_at"] = existing_meta.get("created_at", int(time.time()))
                
                self.redis_client.setex(
                    session_key,
                    Config.SESSION_EXPIRE_TIME,
                    json.dumps(session_meta)
                )
                
            else:
                # 使用备用内存存储
                if session_id not in self.fallback_storage:
                    self.fallback_storage[session_id] = []
                
                self.fallback_storage[session_id].append(conversation_item)
                
                # 限制历史记录长度
                if len(self.fallback_storage[session_id]) > Config.MAX_CONVERSATION_HISTORY:
                    self.fallback_storage[session_id] = self.fallback_storage[session_id][-Config.MAX_CONVERSATION_HISTORY:]
            
            return True
            
        except Exception as e:
            print(f"❌ 添加对话失败: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """获取会话历史"""
        try:
            if self.redis_client:
                # 从Redis获取
                context_key = self._get_context_key(session_id)
                conversations_json = self.redis_client.get(context_key)
                
                if conversations_json:
                    conversations = json.loads(conversations_json)
                else:
                    conversations = []
            else:
                # 从备用存储获取
                conversations = self.fallback_storage.get(session_id, [])
            
            # 应用限制
            if limit:
                conversations = conversations[-limit:]
            
            return conversations
            
        except Exception as e:
            print(f"❌ 获取对话历史失败: {e}")
            return []
    
    def get_context_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        """获取用于LLM的上下文消息列表"""
        try:
            # 获取最近的对话历史
            conversations = self.get_conversation_history(session_id, Config.CONTEXT_WINDOW_SIZE)
            
            messages = []
            for conv in conversations:
                messages.append({"role": "user", "content": conv["question"]})
                messages.append({"role": "assistant", "content": conv["answer"]})
            
            return messages
            
        except Exception as e:
            print(f"❌ 获取LLM上下文失败: {e}")
            return []
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        try:
            if self.redis_client:
                session_key = self._get_session_key(session_id)
                session_meta_json = self.redis_client.get(session_key)
                
                if session_meta_json:
                    session_meta = json.loads(session_meta_json)
                else:
                    session_meta = {}
            else:
                # 从备用存储计算
                conversations = self.fallback_storage.get(session_id, [])
                session_meta = {
                    "conversation_count": len(conversations),
                    "last_activity": conversations[-1]["timestamp"] if conversations else 0,
                    "created_at": conversations[0]["timestamp"] if conversations else 0
                }
            
            # 添加历史记录
            conversations = self.get_conversation_history(session_id)
            session_meta["conversations"] = conversations
            session_meta["session_id"] = session_id
            
            return session_meta
            
        except Exception as e:
            print(f"❌ 获取会话信息失败: {e}")
            return {"session_id": session_id, "conversation_count": 0}
    
    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        try:
            if self.redis_client:
                # 删除Redis中的数据
                session_key = self._get_session_key(session_id)
                context_key = self._get_context_key(session_id)
                
                self.redis_client.delete(session_key)
                self.redis_client.delete(context_key)
            else:
                # 删除备用存储中的数据
                if session_id in self.fallback_storage:
                    del self.fallback_storage[session_id]
            
            print(f"✅ 会话 {session_id} 已清除")
            return True
            
        except Exception as e:
            print(f"❌ 清除会话失败: {e}")
            return False
    
    def list_active_sessions(self) -> List[str]:
        """列出活跃的会话"""
        try:
            if self.redis_client:
                # 从Redis获取所有会话键
                pattern = "session:*"
                keys = self.redis_client.keys(pattern)
                session_ids = [key.replace("session:", "") for key in keys]
            else:
                # 从备用存储获取
                session_ids = list(self.fallback_storage.keys())
            
            return session_ids
            
        except Exception as e:
            print(f"❌ 获取活跃会话失败: {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话（仅针对备用存储）"""
        if self.redis_client:
            # Redis会自动处理过期
            return 0
        
        try:
            current_time = int(time.time())
            expired_sessions = []
            
            for session_id, conversations in self.fallback_storage.items():
                if conversations:
                    last_activity = conversations[-1]["timestamp"]
                    if current_time - last_activity > Config.SESSION_EXPIRE_TIME:
                        expired_sessions.append(session_id)
            
            # 删除过期会话
            for session_id in expired_sessions:
                del self.fallback_storage[session_id]
            
            if expired_sessions:
                print(f"🧹 清理了 {len(expired_sessions)} 个过期会话")
            
            return len(expired_sessions)
            
        except Exception as e:
            print(f"❌ 清理过期会话失败: {e}")
            return 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        status = {
            "redis_connected": self.redis_client is not None,
            "storage_type": "redis" if self.redis_client else "memory",
            "active_sessions": len(self.list_active_sessions())
        }
        
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["redis_status"] = "healthy"
            except:
                status["redis_status"] = "error"
                status["redis_connected"] = False
        
        return status 