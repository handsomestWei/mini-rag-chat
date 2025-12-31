"""
RAG管理模块
负责文档加载、切分、向量库创建和管理
"""

import os
import glob
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from loguru import logger


class RAGManager:
    """RAG知识库管理器"""

    def __init__(self, config, embeddings):
        """
        初始化RAG管理器

        Args:
            config: 配置对象
            embeddings: 嵌入模型
        """
        self.config = config
        self.embeddings = embeddings
        self.vector_store = None

        # 确保目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.config.DATA_PATH,
            self.config.DATA_NEW_PATH
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"目录已确认: {directory}")

    def get_new_documents(self):
        """检查并获取新文档列表（支持PDF和TXT）"""
        new_docs = []
        # PDF文件
        pdf_pattern = os.path.join(self.config.DATA_NEW_PATH, "*.pdf")
        new_docs.extend(glob.glob(pdf_pattern))
        # TXT文件
        txt_pattern = os.path.join(self.config.DATA_NEW_PATH, "*.txt")
        new_docs.extend(glob.glob(txt_pattern))
        return new_docs

    def load_documents_from_path(self, path):
        """
        从指定路径加载文档（支持PDF和TXT）

        Args:
            path: 文档目录路径

        Returns:
            documents: 加载的文档列表
        """
        documents = []

        # 加载PDF文件
        pdf_pattern = os.path.join(path, "**/*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        for pdf_file in pdf_files:
            logger.info(f"📄 加载PDF: {os.path.basename(pdf_file)}")
            try:
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"加载PDF失败: {pdf_file} - {e}")

        # 加载TXT文件
        txt_pattern = os.path.join(path, "**/*.txt")
        txt_files = glob.glob(txt_pattern, recursive=True)
        for txt_file in txt_files:
            logger.info(f"📄 加载TXT: {os.path.basename(txt_file)}")
            try:
                loader = TextLoader(txt_file, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"加载TXT失败: {txt_file} - {e}")

        return documents

    def split_documents(self, documents):
        """
        将文档切分为文本块

        Args:
            documents: 文档列表

        Returns:
            text_chunks: 切分后的文本块列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.debug(f"文档已切分为 {len(text_chunks)} 个块")
        return text_chunks

    def migrate_processed_documents(self):
        """将已处理的文档从data_new移动到data目录"""
        new_docs = self.get_new_documents()

        if not new_docs:
            return 0

        migrated_count = 0
        for doc_path in new_docs:
            doc_name = os.path.basename(doc_path)

            try:
                # 移动到原数据目录
                dest_path = os.path.join(self.config.DATA_PATH, doc_name)
                shutil.move(doc_path, dest_path)
                logger.debug(f"文档已移动到data目录: {doc_name}")

                migrated_count += 1

            except Exception as e:
                logger.error(f"迁移文档失败 {doc_name}: {str(e)}")

        return migrated_count

    def incremental_load(self):
        """
        增量加载新文档到现有向量库

        Returns:
            new_doc_count: 新加载的文档数量
        """
        if self.vector_store is None:
            logger.error("向量库未初始化")
            return 0

        new_docs = self.get_new_documents()

        if not new_docs:
            logger.debug("没有发现新文档")
            return 0

        logger.info(f"📄 发现 {len(new_docs)} 个新文档:")
        for doc in new_docs:
            logger.info(f"  - {os.path.basename(doc)}")

        try:
            # 加载新文档
            logger.info("正在加载新文档...")
            new_documents = self.load_documents_from_path(self.config.DATA_NEW_PATH)
            logger.info(f"已加载 {len(new_documents)} 个新文档页面")

            # 切分文档
            logger.info("正在切分新文档...")
            new_chunks = self.split_documents(new_documents)
            logger.info(f"新文档已切分为 {len(new_chunks)} 个块")

            # 创建新文档的向量并合并到现有向量库
            logger.info("正在将新文档添加到向量库...")
            new_vector_store = FAISS.from_documents(new_chunks, self.embeddings)

            # 合并向量库
            self.vector_store.merge_from(new_vector_store)
            logger.info("向量库合并完成")

            # 保存更新后的向量库
            self.save()

            # 迁移文档
            if self.config.AUTO_MIGRATE_PROCESSED:
                logger.info("正在迁移已处理的文档...")
                migrated = self.migrate_processed_documents()
                logger.info(f"已迁移 {migrated} 个文档")

            return len(new_docs)

        except Exception as e:
            logger.error(f"增量加载失败: {str(e)}", exc_info=True)
            raise

    def create_vector_store(self):
        """创建初始向量库"""
        logger.info("创建向量数据库...")

        # 加载文档
        documents = self.load_documents_from_path(self.config.DATA_PATH)
        logger.info(f"已加载 {len(documents)} 个文档")

        if len(documents) == 0:
            logger.warning("⚠️  未找到任何文档，将创建空的向量库")
            logger.warning(f"💡 请将文档放入目录: {self.config.DATA_PATH}")
            # 创建一个空向量库（使用一个虚拟文档）
            from langchain.schema import Document
            dummy_doc = Document(page_content="初始化向量库", metadata={})
            text_chunks = [dummy_doc]
        else:
            # 切分文档
            text_chunks = self.split_documents(documents)
            logger.info(f"文档已切分为 {len(text_chunks)} 个块")

        # 创建向量库
        logger.info("正在创建向量库...")
        self.vector_store = FAISS.from_documents(text_chunks, self.embeddings)

        # 持久化保存
        self.save()

        return self.vector_store

    def load_vector_store(self):
        """加载已存在的向量库"""
        # 确保向量库目录存在
        Path(self.config.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        
        # 检查向量库文件是否存在（FAISS需要index.faiss和index.pkl）
        index_faiss_path = os.path.join(self.config.VECTOR_STORE_PATH, "index.faiss")
        index_pkl_path = os.path.join(self.config.VECTOR_STORE_PATH, "index.pkl")
        
        if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
            logger.info("向量库文件不存在，将创建新的向量库")
            return self.create_vector_store()

        logger.info("加载已存在的向量数据库...")
        try:
            self.vector_store = FAISS.load_local(
                self.config.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("向量数据库加载完成")
        except Exception as e:
            logger.warning(f"加载向量库失败: {str(e)}，将重新创建向量库")
            return self.create_vector_store()

        # 检查是否有新文档需要增量加载
        if self.config.ENABLE_INCREMENTAL_LOAD:
            logger.debug("检查增量文档...")
            try:
                new_doc_count = self.incremental_load()
                if new_doc_count > 0:
                    logger.info(f"🎉 成功增量加载 {new_doc_count} 个新文档")
            except Exception as e:
                logger.error(f"增量加载出错: {str(e)}")

        return self.vector_store

    def save(self):
        """保存向量库"""
        if self.vector_store is None:
            logger.warning("向量库为空，无法保存")
            return

        self.vector_store.save_local(self.config.VECTOR_STORE_PATH)
        logger.debug(f"向量库已保存到 {self.config.VECTOR_STORE_PATH}")

    def get_retriever(self, search_type="similarity", search_kwargs=None):
        """
        获取检索器

        Args:
            search_type: 检索类型
            search_kwargs: 检索参数

        Returns:
            retriever: 检索器对象
        """
        if self.vector_store is None:
            raise ValueError("向量库未初始化")

        if search_kwargs is None:
            search_kwargs = {
                "k": self.config.RETRIEVER_K,
                "fetch_k": self.config.RETRIEVER_FETCH_K
            }

        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        return retriever

