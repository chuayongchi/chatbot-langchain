from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import torch

app = FastAPI()

# llm = OpenAI(model_name="test-davinci-003")

# Define the Hugging Face pipeline with adjusted parameters
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=300,  # Adjusted to allow input and output to fit
    min_length=50,
    temperature=0.5,
    top_p=0.85,
    top_k=40,
    eos_token_id=50256,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

huggingface_llm = HuggingFacePipeline(pipeline=generator)

# Set up Hugging Face embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use a SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

docs = [
    Document(page_content="LangChain is a framework for building applications with LLMs."),
    Document(page_content="LangChain supports tools, memory, and agents.")
]

# Create FAISS vector store
vector_store = FAISS.from_documents(docs, embedding_model)

# Set up RetrievalQA
retriever = vector_store.as_retriever()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\n\nQuestion: {question}\n\nAnswer the question as accurately as possible."
)

# Configure the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=huggingface_llm,  # Use your text generation pipeline
    chain_type="stuff",  # Alternative: "map_reduce" or "refine"
    retriever=retriever,
    return_source_documents=True  # Include source documents in responses
)

# Test the knowledge retrieval system
# query = "What is LangChain?"
# response = qa_chain.invoke(query)

# # Print the response and source documents
# print(f"Response: {response['result']}")
# for i, doc in enumerate(response['source_documents'], 1):
#     print(f"Source {i}: {doc.page_content}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    response = qa_chain.invoke(request.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)