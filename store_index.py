from src.helper import load_pdf_file, text_split, download_embedding
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embed=download_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "cancerbot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embed
)

