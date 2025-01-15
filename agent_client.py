from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = FastAPI()

# Define the Hugging Face pipeline
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

# Wrap the Hugging Face pipeline in a LangChain-compatible LLM
huggingface_llm = HuggingFacePipeline(pipeline=generator)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["question"],  # Corrected typo: input_variablr -> input_variables
    template="Answer the question as accurately as possible and end with a complete sentence: {question}"  # Improved prompt for clarity
)

# Create the LLMChain
chain = LLMChain(llm=huggingface_llm, prompt=prompt)

# Run the chain with the question
# response = chain.run("What is LangChain?")
# print(response)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    response = chain.run(request.question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
