#!/usr/bin/env python3
"""
智能学习助手 2.0 - 增强版 Conda环境启动脚本
支持Redis会话管理、上下文记忆、多模型AI
"""

import os
import sys
import subprocess
import webbrowser
import time
import socket
from pathlib import Path

def print_header():
    """打印启动头部信息"""
    print("🤖 智能学习助手 2.0 - 增强版")
    print("=" * 60)
    print("🆕 新功能:")
    print("   • Redis会话管理 - 支持持久化对话")
    print("   • 上下文记忆 - 智能多轮对话")
    print("   • 双模型支持 - OpenAI + DeepSeek")
    print("   • 会话监控 - 实时状态管理")
    print("=" * 60)

def check_conda():
    """检查conda是否可用"""
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Conda版本: {result.stdout.strip()}")
            return True
        else:
            print("❌ Conda未找到")
            return False
    except FileNotFoundError:
        print("❌ Conda未安装或不在PATH中")
        return False

def check_redis():
    """检查Redis是否可用"""
    print("\n🗃️  检查Redis服务...")
    
    # 首先检查Redis服务是否可以连接（最重要的检查）
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)  # 增加超时时间
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        
        if result == 0:
            print("✅ Redis服务正在运行 (localhost:6379)")
            
            # 尝试获取Redis版本信息（可选）
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                info = r.info()
                version = info.get('redis_version', 'unknown')
                print(f"   版本: {version}")
                r.close()
            except:
                print("   (Docker或远程Redis实例)")
            
            return True
        else:
            print("⚠️  无法连接到Redis服务 (localhost:6379)")
            start_redis_hint()
            return False
    except Exception as e:
        print(f"❌ Redis连接检查失败: {e}")
        start_redis_hint()
        return False

def install_redis_hint():
    """提供Redis安装提示"""
    print("\n💡 Redis安装指南:")
    if os.name == 'nt':  # Windows
        print("   方法1 - 使用Windows Subsystem for Linux (WSL):")
        print("     wsl --install")
        print("     wsl")
        print("     sudo apt update")
        print("     sudo apt install redis-server")
        print("     sudo service redis-server start")
        print("\n   方法2 - 使用Docker:")
        print("     docker run -d -p 6379:6379 redis:latest")
    else:  # Linux/Mac
        print("   Ubuntu/Debian:")
        print("     sudo apt update")
        print("     sudo apt install redis-server")
        print("     sudo systemctl start redis-server")
        print("\n   CentOS/RHEL:")
        print("     sudo yum install redis")
        print("     sudo systemctl start redis")
        print("\n   macOS (使用Homebrew):")
        print("     brew install redis")
        print("     brew services start redis")
        print("\n   Docker (通用):")
        print("     docker run -d -p 6379:6379 redis:latest")

def start_redis_hint():
    """提供Redis启动提示"""
    print("\n💡 Redis启动指南:")
    if os.name == 'nt':  # Windows
        print("   如果使用WSL:")
        print("     wsl")
        print("     sudo service redis-server start")
        print("\n   如果使用Docker:")
        print("     docker run -d -p 6379:6379 redis:latest")
    else:  # Linux/Mac
        print("   Linux:")
        print("     sudo systemctl start redis-server")
        print("     # 或者")
        print("     redis-server")
        print("\n   macOS:")
        print("     brew services start redis")
        print("     # 或者")
        print("     redis-server")
        print("\n   Docker:")
        print("     docker run -d -p 6379:6379 redis:latest")

def check_environment():
    """检查conda环境是否存在"""
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True
        )
        
        if "intelligent-study-assistant" in result.stdout:
            print("✅ Conda环境已存在")
            return True
        else:
            print("⚠️  Conda环境不存在")
            return False
    except Exception as e:
        print(f"❌ 检查环境失败: {e}")
        return False

def create_environment():
    """创建conda环境"""
    print("\n📦 正在创建conda环境...")
    
    if not Path("environment.yml").exists():
        print("❌ environment.yml文件不存在")
        print("💡 正在创建environment.yml文件...")
        create_environment_yml()
    
    try:
        # 创建环境
        subprocess.run(
            ["conda", "env", "create", "-f", "environment.yml"],
            check=True
        )
        print("✅ Conda环境创建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 环境创建失败: {e}")
        print("💡 尝试手动运行: conda env create -f environment.yml")
        return False

def create_environment_yml():
    """创建environment.yml文件"""
    env_content = """name: intelligent-study-assistant
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
    - fastapi==0.104.1
    - uvicorn==0.24.0
    - langchain==0.1.0
    - langchain-openai==0.0.2
    - langchain-community==0.0.10
    - chromadb==0.4.18
    - pypdf2==3.0.1
    - python-docx==1.1.0
    - python-multipart==0.0.6
    - pydantic==2.5.1
    - tiktoken==0.5.2
    - openai==1.93.0
    - requests==2.31.0
    - numpy==1.24.3
    - pandas==2.0.3
    - sentence-transformers==2.2.2
    - python-dotenv==1.0.0
    - redis==5.0.1
    - aioredis==2.0.1
"""
    
    with open("environment.yml", "w", encoding='utf-8') as f:
        f.write(env_content)
    print("✅ environment.yml文件创建完成")

def install_dependencies():
    """安装Python依赖"""
    print("\n📦 正在安装/更新依赖包...")
    
    # 确保requirements.txt存在且包含Redis
    if not Path("requirements.txt").exists():
        print("⚠️  requirements.txt不存在，正在创建...")
        create_requirements_txt()
    
    try:
        # 在conda环境中安装依赖
        if os.name == 'nt':
            subprocess.run([
                "conda", "run", "-n", "intelligent-study-assistant",
                "pip", "install", "-r", "requirements.txt", "--upgrade"
            ], check=True)
        else:
            subprocess.run([
                "conda", "run", "-n", "intelligent-study-assistant",
                "pip", "install", "-r", "requirements.txt", "--upgrade"
            ], check=True)
        
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def create_requirements_txt():
    """创建requirements.txt文件"""
    requirements_content = """# 智能学习助手 2.0 - Python依赖
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.18
pypdf2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
pydantic==2.5.1
tiktoken==0.5.2
openai==1.93.0
requests==2.31.0
numpy==1.24.3
pandas==2.0.3
sentence-transformers==2.2.2
python-dotenv==1.0.0
redis==5.0.1
aioredis==2.0.1
"""
    
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write(requirements_content)
    print("✅ requirements.txt文件创建完成")

def activate_and_run():
    """激活环境并运行后端"""
    print("\n🚀 正在激活环境并启动服务...")
    
    # 根据操作系统选择激活命令
    if os.name == 'nt':  # Windows
        script_content = """@echo off
echo 🔄 激活conda环境...
call conda activate intelligent-study-assistant
if errorlevel 1 (
    echo ❌ 环境激活失败
    pause
    exit /b 1
)

echo 📂 切换到后端目录...
cd backend
if errorlevel 1 (
    echo ❌ 后端目录不存在
    pause
    exit /b 1
)

echo 🚀 启动智能学习助手服务...
echo.
echo 📡 服务地址: http://localhost:8000
echo 📚 API文档: http://localhost:8000/docs  
echo 🗃️  Redis状态: 自动检测
echo.
echo 按 Ctrl+C 停止服务
echo.
python main.py
pause
"""
        script_path = "run_backend.bat"
    else:
        script_content = """#!/bin/bash
echo "🔄 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate intelligent-study-assistant

if [ $? -ne 0 ]; then
    echo "❌ 环境激活失败"
    exit 1
fi

echo "📂 切换到后端目录..."
cd backend

if [ $? -ne 0 ]; then
    echo "❌ 后端目录不存在"
    exit 1
fi

echo "🚀 启动智能学习助手服务..."
echo
echo "📡 服务地址: http://localhost:8000"
echo "📚 API文档: http://localhost:8000/docs"
echo "🗃️  Redis状态: 自动检测"
echo
echo "按 Ctrl+C 停止服务"
echo
python main.py
"""
        script_path = "run_backend.sh"
    
    # 写入脚本文件
    with open(script_path, "w", encoding='utf-8') as f:
        f.write(script_content)
    
    # 在Windows上运行bat文件，在Linux/Mac上运行sh文件
    if os.name == 'nt':
        try:
            process = subprocess.Popen([script_path], shell=True)
            print("✅ 后端服务启动中...")
            print("📡 后端地址: http://localhost:8000")
            print("📚 API文档: http://localhost:8000/docs")
            return process
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return None
    else:
        # 给脚本执行权限
        os.chmod(script_path, 0o755)
        try:
            process = subprocess.Popen([f"./{script_path}"])
            print("✅ 后端服务启动中...")
            print("📡 后端地址: http://localhost:8000")
            print("📚 API文档: http://localhost:8000/docs")
            return process
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return None

def create_directories():
    """创建必要的目录"""
    directories = ["uploads", "vector_store", "backend/uploads", "backend/vector_store"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ 目录结构创建完成")

def wait_for_backend():
    """等待后端服务启动"""
    print("\n⏳ 等待后端服务启动...")
    max_attempts = 30
    
    for i in range(max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            
            if result == 0:
                print("✅ 后端服务已启动")
                return True
            else:
                print(f"   尝试 {i+1}/{max_attempts}...")
                time.sleep(1)
        except Exception:
            time.sleep(1)
    
    print("⚠️  后端服务启动超时，请手动检查")
    return False

def open_frontend():
    """打开前端页面"""
    print("\n🌐 正在打开前端页面...")
    frontend_path = Path("frontend/index.html")
    
    if not frontend_path.exists():
        print("❌ 前端文件不存在")
        return False
    
    try:
        frontend_url = f"file://{frontend_path.absolute()}"
        webbrowser.open(frontend_url)
        print("✅ 前端页面已打开")
        print("🌐 前端地址:", frontend_url)
        return True
    except Exception as e:
        print(f"❌ 前端打开失败: {e}")
        print(f"💡 请手动打开: {frontend_path.absolute()}")
        return False

def print_final_info(redis_available):
    """打印最终信息"""
    print("\n🎉 智能学习助手 2.0 启动完成！")
    print("=" * 60)
    print("🌟 新功能体验:")
    print("  • 会话管理: 左侧边栏可查看和切换对话")
    print("  • 上下文记忆: AI会记住之前的对话内容")
    print("  • 多模型选择: 可选择OpenAI或DeepSeek模型")
    print("  • 服务监控: 实时查看系统运行状态")
    
    if redis_available:
        print("\n🗃️  Redis状态: ✅ 已连接 (支持持久化会话)")
    else:
        print("\n🗃️  Redis状态: ⚠️  未连接 (使用内存存储)")
        print("       会话数据在服务重启后会丢失")
    
    print("\n📖 使用说明:")
    print("1. 上传PDF、TXT或DOCX文件到系统")
    print("2. 在智能问答中开始对话，体验上下文记忆")
    print("3. 尝试生成总结、练习题和学习计划")
    print("4. 查看左侧会话管理，切换不同对话")
    print("5. 监控服务状态和会话统计")
    
    print("\n🛠️  管理命令:")
    print("• 停止服务: 按 Ctrl+C")
    print("• 重启服务: 重新运行此脚本")
    print("• 查看日志: 检查终端输出")
    print("• 清理会话: 使用前端界面的清理按钮")

def main():
    """主函数"""
    print_header()
    
    # 检查conda
    if not check_conda():
        print("💡 请安装Anaconda或Miniconda")
        print("   下载地址: https://www.anaconda.com/products/distribution")
        return
    
    # 检查Redis（不强制要求）
    redis_available = check_redis()
    
    # 检查环境
    if not check_environment():
        user_input = input("\n是否创建新的conda环境？(y/n): ")
        if user_input.lower() in ['y', 'yes', '是']:
            if not create_environment():
                return
        else:
            print("❌ 需要创建conda环境才能继续")
            return
    
    # 安装/更新依赖
    if not install_dependencies():
        print("⚠️  依赖安装失败，但可以尝试继续运行")
    
    # 创建目录
    create_directories()
    
    # 启动后端
    backend_process = activate_and_run()
    if not backend_process:
        return
    
    # 等待后端启动
    if wait_for_backend():
        # 打开前端
        open_frontend()
        
        # 显示最终信息
        print_final_info(redis_available)
    else:
        print("❌ 后端服务启动失败")
        return
    
    try:
        # 保持脚本运行
        print("\n📱 服务正在运行...")
        print("   按 Ctrl+C 停止所有服务")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        backend_process.terminate()
        print("✅ 服务已停止")

if __name__ == "__main__":
    main() 