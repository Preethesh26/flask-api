from flask import Flask, request, jsonify
import os
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# Paths
dataset_path = "plants.xlsx"
chroma_db_dir = "chroma_db"

# Load and split the dataset
def load_dataset():
    df = pd.read_excel(dataset_path)
    documents = []
    for _, row in df.iterrows():
        text_parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                if isinstance(row[col], (str, int, float)):
                    text_parts.append(f"{col}: {row[col]}")
        text = "\n".join(text_parts)
        doc = Document(page_content=text, metadata={"source": "plants.xlsx"})
        documents.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Create the vector database
def create_vector_db(texts):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(chroma_db_dir):
        os.makedirs(chroma_db_dir)
    vectordb = Chroma.from_documents(texts, embedding, persist_directory=chroma_db_dir)
    vectordb.persist()
    return vectordb

# Load texts and initialize DB only once
texts = load_dataset()
vectordb = create_vector_db(texts)

# Answer question
def answer_question(query):
    if not os.path.exists(chroma_db_dir):
        return {"error": "Vector database not found."}

    docs = vectordb.similarity_search(query, k=3)
    if not docs:
        return {"error": "No relevant information found."}

    info = []
    preparation_methods = set()
    unique_responses = set()
    location_link = "https://maps.app.goo.gl/cPghnzW23zHydnGq7?g_st=aw"

    for doc in docs:
        content = doc.page_content.strip()
        if content not in unique_responses:
            unique_responses.add(content)
            lines = content.split("\n")
            for line in lines:
                if not line.strip():
                    continue
                if "Preparation Method" in line:
                    try:
                        preparation = line.strip().split(":", 1)[1].strip()
                        preparation_methods.add(preparation)
                    except IndexError:
                        continue
                elif ":" in line:
                    try:
                        key, value = line.split(":", 1)
                        info.append({key.strip(): value.strip()})
                    except ValueError:
                        continue

    return {
        "info": info,
        "preparation_method": list(preparation_methods),
        "location": location_link
    }

@app.route('/ask', methods=['GET'])
def ask():
    question = request.args.get('question')  # get question from query params
    if question:
        try:
            answer = answer_question(question)  # this should call your RAG function
            return jsonify({'question': question, 'answer': answer})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Please provide a question using the "question" query parameter.'}), 400

@app.route('/')
def home():
    return "✅ RAG API is running. Use /ask?question=YourQuestion to get answers."

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # Ensure it's publicly accessible
