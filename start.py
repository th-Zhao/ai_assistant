#!/usr/bin/env python3
"""
智能学习助手启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"   当前版本：{sys.version}")
        return False
    print(f"✅ Python版本检查通过：{sys.version.split()[0]}")
    return True

def install_dependencies():
    """安装依赖"""
    print("\n📦 正在安装依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败：{e}")
        print("💡 尝试使用清华源：")
        print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False

def create_directories():
    """创建必要的目录"""
    directories = ["uploads", "vector_store", "backend/uploads", "backend/vector_store"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ 目录创建完成")

def start_backend():
    """启动后端服务"""
    print("\n🚀 正在启动后端服务...")
    backend_path = Path("backend")
    
    if not backend_path.exists():
        print("❌ 后端目录不存在")
        return None
    
    # 切换到backend目录并启动服务
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_path,
            env=env
        )
        print("✅ 后端服务启动成功")
        print("📡 后端地址：http://localhost:8000")
        print("📚 API文档：http://localhost:8000/docs")
        return process
    except Exception as e:
        print(f"❌ 后端启动失败：{e}")
        return None

def open_frontend():
    """打开前端页面"""
    print("\n🌐 正在打开前端页面...")
    frontend_path = Path("frontend/index.html")
    
    if not frontend_path.exists():
        print("❌ 前端文件不存在")
        return False
    
    try:
        # 获取绝对路径
        frontend_url = f"file://{frontend_path.absolute()}"
        webbrowser.open(frontend_url)
        print("✅ 前端页面已打开")
        print("🌐 前端地址：", frontend_url)
        return True
    except Exception as e:
        print(f"❌ 前端打开失败：{e}")
        print(f"💡 请手动打开：{frontend_path.absolute()}")
        return False

def main():
    """主函数"""
    print("🎓 智能学习助手启动器")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检查依赖文件
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt 文件不存在")
        return
    
    # 安装依赖
    install_dependencies()
    
    # 创建目录
    create_directories()
    
    # 启动后端
    backend_process = start_backend()
    if not backend_process:
        return
    
    # 等待后端启动
    print("⏳ 等待后端服务启动...")
    time.sleep(3)
    
    # 打开前端
    open_frontend()
    
    print("\n🎉 启动完成！")
    print("=" * 50)
    print("📖 使用说明：")
    print("1. 在网页中上传PDF、TXT或DOCX文件")
    print("2. 点击相应功能按钮体验智能学习助手")
    print("3. 按Ctrl+C停止服务")
    print("=" * 50)
    
    try:
        # 等待用户中断
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 正在关闭服务...")
        backend_process.terminate()
        print("✅ 服务已关闭")

if __name__ == "__main__":
    main() 