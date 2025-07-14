# main.py

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "chroma_db"

def main():
    print("Loading vector store...")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=OpenAIEmbeddings())

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    while True:
        query = input("\n🧠 Ask your research question (or 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa(query)
        print("\n📄 Answer:")
        print(result["result"])
        print("\n📚 Sources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source"))

if __name__ == "__main__":
    main()
