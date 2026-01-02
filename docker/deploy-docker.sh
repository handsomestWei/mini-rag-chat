#!/bin/bash

# Mini RAG Chat Docker 部署脚本

set -e

echo "🚀 开始部署 Mini RAG Chat 系统..."

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Docker 和 Docker Compose
check_dependencies() {
    echo -e "${YELLOW}检查依赖...${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装，请先安装 Docker${NC}"
        exit 1
    fi

    # 检查 docker compose（新版本）或 docker-compose（旧版本）
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
        echo -e "${GREEN}✅ 使用 docker compose（新版本）${NC}"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        echo -e "${GREEN}✅ 使用 docker-compose（旧版本）${NC}"
    else
        echo -e "${RED}❌ Docker Compose 未安装，请先安装 Docker Compose${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ 依赖检查通过${NC}"
}

# 创建必要的目录
create_directories() {
    echo -e "${YELLOW}创建必要的目录...${NC}"

    # 创建主目录
    BASE_DIR="/opt/mini-rag"
    echo -e "${YELLOW}创建主目录: ${BASE_DIR}${NC}"

    sudo mkdir -p "$BASE_DIR"
    sudo chown $(whoami):$(whoami) "$BASE_DIR"

    # 创建子目录
    echo -e "${YELLOW}创建子目录...${NC}"

    DIRS=(
        "$BASE_DIR/data"                    # 知识库数据
        "$BASE_DIR/data_new"               # 新数据目录
        "$BASE_DIR/vector_store"           # 向量存储
        "$BASE_DIR/model"                  # 模型目录
        "$BASE_DIR/intent_fine_tuning"     # 意图识别模型
        "$BASE_DIR/templates"              # 前端模板目录
        "$BASE_DIR/log"                    # 日志目录（与docker-compose.yml一致）
        "$BASE_DIR/ollama/data"            # Ollama 数据目录
        "$BASE_DIR/ollama/model"           # Ollama 模型目录
        "$BASE_DIR/ssl"                    # SSL证书目录
    )

    for dir in "${DIRS[@]}"; do
        sudo mkdir -p "$dir"
        sudo chown $(whoami):$(whoami) "$dir"
        sudo chmod 755 "$dir"
        echo -e "${GREEN}✅ 创建目录: $dir${NC}"
    done

    # 设置权限
    echo -e "${YELLOW}设置目录权限...${NC}"

    # 只读目录
    sudo chmod 755 "$BASE_DIR/data"
    sudo chmod 755 "$BASE_DIR/model"
    sudo chmod 755 "$BASE_DIR/intent_fine_tuning"

    # 读写目录
    sudo chmod 777 "$BASE_DIR/data_new"
    sudo chmod 777 "$BASE_DIR/vector_store"
    sudo chmod 777 "$BASE_DIR/log"
    sudo chmod 777 "$BASE_DIR/ollama/data"

    echo -e "${GREEN}✅ 目录创建和权限设置完成${NC}"
}

# 检查模型文件
check_models() {
    echo -e "${YELLOW}检查模型文件...${NC}"

    BASE_DIR="/opt/mini-rag"

    # 检查本地模型文件
    if [ ! -d "model" ]; then
        echo -e "${RED}❌ 本地 model 目录不存在，请确保模型文件已下载${NC}"
        exit 1
    fi

    if [ ! -d "data" ]; then
        echo -e "${RED}❌ 本地 data 目录不存在，请确保知识库文件已准备${NC}"
        exit 1
    fi

    # 复制文件到绝对路径目录
    echo -e "${YELLOW}复制文件到部署目录...${NC}"

    # 复制模型文件
    if [ -d "model" ] && [ "$(ls -A model)" ]; then
        echo "复制模型文件到 $BASE_DIR/model/"
        sudo cp -r ./model/* "$BASE_DIR/model/" 2>/dev/null || true
        echo -e "${GREEN}✅ 模型文件复制完成${NC}"
    fi

    # 复制知识库文件
    if [ -d "data" ] && [ "$(ls -A data)" ]; then
        echo "复制知识库文件到 $BASE_DIR/data/"
        sudo cp -r ./data/* "$BASE_DIR/data/" 2>/dev/null || true
        echo -e "${GREEN}✅ 知识库文件复制完成${NC}"
    fi

    # 复制意图识别模型
    if [ -d "intent_fine_tuning" ] && [ "$(ls -A intent_fine_tuning)" ]; then
        echo "复制意图识别模型到 $BASE_DIR/intent_fine_tuning/"
        sudo cp -r ./intent_fine_tuning/* "$BASE_DIR/intent_fine_tuning/" 2>/dev/null || true
        echo -e "${GREEN}✅ 意图识别模型复制完成${NC}"
    fi

    # 复制前端模板文件
    if [ -d "templates" ] && [ "$(ls -A templates)" ]; then
        echo "复制前端模板文件到 $BASE_DIR/templates/"
        sudo cp -r ./templates/* "$BASE_DIR/templates/" 2>/dev/null || true
        echo -e "${GREEN}✅ 前端模板文件复制完成${NC}"
    fi

    # 复制配置文件
    if [ -f "config.yaml" ]; then
        echo "复制配置文件到 $BASE_DIR/config.yaml"
        sudo cp ./config.yaml "$BASE_DIR/config.yaml" 2>/dev/null || true
        echo -e "${GREEN}✅ 配置文件复制完成${NC}"
    elif [ -f "config.yaml.example" ]; then
        echo "复制配置文件模板到 $BASE_DIR/config.yaml"
        sudo cp ./config.yaml.example "$BASE_DIR/config.yaml" 2>/dev/null || true
        echo -e "${YELLOW}⚠️  使用配置文件模板，请根据需要修改${NC}"
    else
        echo -e "${YELLOW}⚠️  未找到 config.yaml 文件，将使用容器内的默认配置${NC}"
    fi

    echo -e "${GREEN}✅ 模型和知识库检查通过${NC}"
}

# 拉取 Ollama 模型
pull_ollama_models() {
    echo -e "${YELLOW}拉取 Ollama 模型...${NC}"

    # 启动 Ollama 服务
    $COMPOSE_CMD up -d ollama

    # 等待 Ollama 启动
    echo "等待 Ollama 服务启动..."
    sleep 10

    # 检查容器是否启动
    if ! docker ps --format "{{.Names}}" | grep -q "^mini-rag-ollama$"; then
        echo -e "${RED}❌ Ollama 容器启动失败${NC}"
        docker logs mini-rag-ollama 2>/dev/null || echo "无法获取容器日志"
        return 1
    fi

    # 拉取 Qwen2 模型
    echo "拉取 Qwen2:1.5b 模型..."
    if docker exec mini-rag-ollama ollama pull qwen2:1.5b; then
        echo -e "${GREEN}✅ Ollama 模型拉取完成${NC}"
    else
        echo -e "${YELLOW}⚠️  Ollama 模型拉取失败，可能网络问题，请稍后手动拉取${NC}"
    fi
}

# 构建和启动服务
deploy_services() {
    echo -e "${YELLOW}构建和启动服务...${NC}"

    # 构建镜像
    $COMPOSE_CMD build

    # 启动所有服务
    $COMPOSE_CMD up -d

    echo -e "${GREEN}✅ 服务启动完成${NC}"
}

# 检查服务状态
check_services() {
    echo -e "${YELLOW}检查服务状态...${NC}"

    sleep 15

    # 检查 Ollama 服务
    if docker ps --format "{{.Names}}" | grep -q "^mini-rag-ollama$"; then
        if docker exec mini-rag-ollama curl -f http://localhost:11434/api/tags &> /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama 服务正常${NC}"
        else
            echo -e "${YELLOW}⚠️  Ollama 服务启动中或异常，请稍后检查${NC}"
        fi
    else
        echo -e "${RED}❌ Ollama 容器未运行${NC}"
    fi

    # 检查 Mini RAG 服务
    if docker ps --format "{{.Names}}" | grep -q "^mini-rag-chat$"; then
        if curl -f http://localhost:5000/health &> /dev/null 2>&1; then
            echo -e "${GREEN}✅ Mini RAG 服务正常${NC}"
        else
            echo -e "${YELLOW}⚠️  Mini RAG 服务启动中或异常，请稍后检查${NC}"
            echo -e "${YELLOW}   查看日志: docker logs mini-rag-chat${NC}"
        fi
    else
        echo -e "${RED}❌ Mini RAG 容器未运行${NC}"
    fi

    # 检查 Nginx 服务（如果启用）
    if docker ps --format "{{.Names}}" | grep -q "mini-rag-nginx"; then
        if curl -f http://localhost/health &> /dev/null 2>&1; then
            echo -e "${GREEN}✅ Nginx 服务正常${NC}"
        else
            echo -e "${YELLOW}⚠️  Nginx 服务异常${NC}"
        fi
    else
        echo -e "${YELLOW}ℹ️  Nginx 服务未启用${NC}"
    fi

    # 显示容器状态
    echo -e "${YELLOW}容器状态:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep mini-rag || echo "未找到相关容器"
}

# 显示访问信息
show_access_info() {
    echo -e "${GREEN}"
    echo "🎉 部署完成！"
    echo "================================"
    echo "访问地址："
    echo "  • 主应用: http://localhost:5000"
    if docker ps --format "table {{.Names}}" | grep -q "mini-rag-nginx"; then
        echo "  • Nginx 代理: http://localhost"
    fi
    echo ""
    echo "数据目录："
    echo "  • 主目录: /opt/mini-rag/"
    echo "  • 知识库: /opt/mini-rag/data/"
    echo "  • 模型: /opt/mini-rag/model/"
    echo "  • 向量库: /opt/mini-rag/vector_store/"
    echo "  • 前端模板: /opt/mini-rag/templates/"
    echo "  • 日志: /opt/mini-rag/log/"
    echo ""
    echo "管理命令："
    echo "  • 查看日志: ${COMPOSE_CMD:-docker-compose} logs -f"
    echo "  • 停止服务: ${COMPOSE_CMD:-docker-compose} down"
    echo "  • 重启服务: ${COMPOSE_CMD:-docker-compose} restart"
    echo "  • 更新服务: ${COMPOSE_CMD:-docker-compose} pull && ${COMPOSE_CMD:-docker-compose} up -d"
    echo "  • 查看状态: docker ps | grep mini-rag"
    echo "================================"
    echo -e "${NC}"
}

# 主函数
main() {
    check_dependencies
    create_directories
    check_models
    pull_ollama_models
    deploy_services
    check_services
    show_access_info
}

# 运行主函数
main "$@"
