from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from typing_extensions import List, TypedDict
from pinecone import ServerlessSpec
from IPython.display import Image, display
# Load environment variables from .env file
#env_path = Path(__file__).parent / "keys.env"
load_dotenv("keys.env")
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)

index_name = "arkana"  # must be lowercase for Pinecone
# if pc.has_index(index_name):
#     pc.delete_index(index_name)

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Load and chunk contents of the blog
file_path = "./manuals/692850638-ARKANA-vehicle-user-manual.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks in batches to avoid Pinecone 2MB request limit
batch_size = 50
for i in range(0, len(all_splits), batch_size):
    batch = all_splits[i:i+batch_size]
    vector_store.add_documents(documents=batch)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
result = graph.invoke({"question":"What kind problems can happen in the vehicle?" })

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')