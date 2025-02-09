import os
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import pandas as pd
import fitz  # PyMuPDF для работы с PDF
from docx import Document
import logging

# Конфигурация
llm_model = "llama3.2"
base_url = "http://localhost:11434"  # Настройте URL для вашего Ollama-сервера

# Инициализация ChromaDB клиента
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Логирование
logging.basicConfig(filename="query_logs.log", level=logging.INFO)

# Пользовательская функция эмбеддингов для ChromaDB
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Создание функции эмбеддингов
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(model=llm_model, base_url=base_url)
)

# Создание или получение коллекции
collection_name = "constitution_collection"
constitution_collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "Коллекция Конституции Казахстана"},
    embedding_function=embedding
)

# Функция для добавления документов в коллекцию
def add_documents_to_collection(collection, documents, ids):
    collection.add(documents=documents, ids=ids)

# Функция для обработки текста Конституции
def preprocess_constitution(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    articles = content.split("Статья")  # Обновлено для текста на русском
    preprocessed = {f"Статья {i}": text.strip() for i, text in enumerate(articles, 1) if text.strip()}
    return preprocessed

# Функция для выполнения запроса к ChromaDB
def query_chromadb(collection, query_text, n_results=5):  # Увеличено количество результатов
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    print("Метаданные:", results.get("metadatas", []))  # Добавлен вывод метаданных для отладки
    return results["documents"], results.get("metadatas", [])

# Функция взаимодействия с Ollama LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model, base_url=base_url)
    return llm.stream(prompt)

# Основной конвейер RAG
def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(constitution_collection, query_text)
    context = "\n\n".join([doc for docs in retrieved_docs for doc in docs]) if retrieved_docs else "Релевантные документы не найдены."
    articles = ", ".join([meta.get("id", "N/A") for meta_list in metadata for meta in (meta_list if isinstance(meta_list, list) else [meta_list]) if isinstance(meta, dict) and meta.get("id")]) or "Не указаны"
    augmented_prompt = f"Контекст: {context}\n\nВопрос: {query_text}\n\nОтвет с указанием статей:"
    response = query_ollama(augmented_prompt)
    full_response = "".join(response)
    return full_response + f"\n\nУпомянутые статьи: {articles}"

# Интерфейс Streamlit
st.title("Чат-бот по Конституции Казахстана")

# Предзагрузка Конституции
constitution_file_path = "constitution_kazakhstan.txt"
if os.path.exists(constitution_file_path):
    st.info("Загрузка текста Конституции...")
    constitution_content = preprocess_constitution(constitution_file_path)
    for article, content in constitution_content.items():
        add_documents_to_collection(constitution_collection, [content], [article])
    st.success("Текст Конституции успешно загружен!")

# Загрузка пользовательских документов
uploaded_files = st.file_uploader(
    "Загрузите ваши документы (.txt, .pdf, .docx, .csv)", 
    type=["txt", "pdf", "docx", "csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = uploaded_file.name
        if file_id.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
        elif file_id.endswith(".pdf"):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                content = "".join([page.get_text() for page in pdf])
        elif file_id.endswith(".docx"):
            doc = Document(uploaded_file)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif file_id.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        add_documents_to_collection(constitution_collection, [content], [file_id])
        st.success(f"{uploaded_file.name} добавлен в коллекцию!")

# Чат
prompt = st.text_area("Введите ваш запрос:")

if st.button("Сгенерировать"):
    if prompt:
        with st.spinner("Генерация ответа..."):
            logging.info(f"Пользовательский запрос: {prompt}")
            response = rag_pipeline(prompt)
            st.write("### Ответ:")
            st.write(response)
