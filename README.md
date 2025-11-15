# AmbedkarGPT – AI Intern Assignment (Kalpit Pvt Ltd)

This project is a simple **Retrieval-Augmented Generation (RAG)** system built as part of the **AI Intern Assignment** for Kalpit Pvt Ltd (UK).

The system loads a speech by Dr. B. R. Ambedkar, creates vector embeddings using **HuggingFace + ChromaDB**, and answers user questions using **Ollama (Mistral 7B)** through a command-line interface.

---

## Features

- Loads text from `speech.txt`
- Splits text into small overlapping chunks
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in a local **ChromaDB** vector store
- Retrieves relevant context using semantic similarity
- Uses **Ollama's Mistral model** to generate answers
- Fully offline and free to run
- Command-line Q&A interface

---

## Project Structure

- main.py
- requirements.txt
- speech.txt 
- chroma_db/ # auto-generated after the first run


---

## Requirements

- Python 3.8+
- macOS / Linux / Windows (works best with macOS + Ollama)
- Ollama with **Mistral 7B** installed

---

## Installation & Setup

### **1. Install Ollama**
Download Ollama (Mac/Linux) from:

https://ollama.com/download

Then pull the Mistral model:

ollama pull mistral

### **2. Create Virtual environment**

python3 -m venv venv
source venv/bin/activate   # macOS/Linux

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the Application

python main.py





