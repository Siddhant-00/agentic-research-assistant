from dotenv import load_dotenv

from langchain_community.document_loaders import (
    WebBaseLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

load_dotenv()

SOURCE_MAP = {
    "https://lilianweng.github.io/posts/2023-06-23-agent/":
        "Lilian Weng - AI Agents",

    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/":
        "Lilian Weng - Prompt Engineering",

    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/":
        "Lilian Weng - Adversarial Attacks"
}

urls = list(
    SOURCE_MAP.keys()
)

documents = []

for url in urls:

    loader = WebBaseLoader(url)

    docs = loader.load()

    for doc in docs:

        doc.metadata["source"] = SOURCE_MAP[url]

        doc.metadata["url"] = url

        documents.append(doc)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(
    "faiss_index"
)

print(
    f"Indexed {len(chunks)} chunks"
)