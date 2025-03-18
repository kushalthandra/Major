from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Hardcoded Mistral API key
MISTRAL_API_KEY = "xh304k1RPHj82gpPeJyj47dSJ5xNkx1C"  # ⚠️ Hardcoded key

# Load embedding model
embed = download_embedding()

# Define Pinecone index
index_name = "cancerbot"

# Initialize retriever
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embed
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize LLM
llm = ChatMistralAI(
    temperature=0.2,
    max_tokens=500,
    api_key=MISTRAL_API_KEY  # Using the hardcoded key
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}")
    ]
)

# Create the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        msg = data.get("msg", "").strip()
        
        if not msg:
            return jsonify({"answer": "Please enter a valid message."}), 400

        print("User Input:", msg)
        response = rag_chain.invoke({"input": msg})
        
        print("Response:", response["answer"])
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"answer": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(host="10.0.0.0", port=8080, debug=True)
