from src.helper import load_pdf, text_split, download_huggingface_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()


Pinecone(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_API_ENV'))

index_name="llama2-medical-chatbot"

#create embeddings for the text chunks and store them in pinecone
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)