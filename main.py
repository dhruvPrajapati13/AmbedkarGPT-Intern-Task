import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA

# Configuration
Embedding_Model = "sentence-transformers/all-MiniLM-L6-v2"
LLM_Model = "mistral"
Vector_DB = "chroma_db"

# RAG Pipeline
def setup_rag_pipeline():

    #1. Loading Speech.txt
    print("1. Loading text from")
    loader = TextLoader("speech.txt")
    documents = loader.load()
    
    # 2. Split the text into chunks
    print("2. Splitting text into chunks.")
    text_splitter = CharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings and store them in vector store (ChromaDB)
    embeddings = HuggingFaceEmbeddings(model_name=Embedding_Model)

    print("3.Creating and persisting ChromaDB vector store")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=Vector_DB
    )
    print("ChromaDB setup complete.Documents stored and indexed.")

    #4. Initialize Ollama LLM
    print("4. Initializing LLM")
    llm = Ollama(model=LLM_Model)
    
    # 5. Create the RetrievalQA Chain
    # This chain automatically handles retrieval
    print("5. Creating Retrieval QA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' uses all retrieved documents as context
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    
    print("--- RAG Pipeline Ready ---")
    return qa_chain

def run_qa_system(qa_chain):
    print("\n\n--- Start Q&A System ---")
    print("Ask a question about the speech (type 'exit' to quit).")

    while True:
        try:
            question = input("\nYour Question: ")
            
            if question.lower() == 'exit':
                print("Exiting Q&A system. Goodbye!")
                break
            
            if not question.strip():
                continue

            # Generate the answer
            print("Generating answer...")
            result = qa_chain({"query": question})
            
            # Print the result
            print("\n--- ANSWER ---")
            print(result['result'])
            print("--------------")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure Ollama is running and the 'mistral' model is pulled.")
            break

if __name__ == "__main__":
    if not os.path.exists(Vector_DB):
        print("\nNOTE: The ChromaDB directory does not exist. It will be created on first run.")
    
    # Set up and run the system
    qa_chain = setup_rag_pipeline()
    run_qa_system(qa_chain)


