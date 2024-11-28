import os
import faiss
import tensorflow_hub as hub
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests

# Load Universal Sentence Encoder (USE)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from a website
def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join([p.text for p in soup.find_all('p')])


# Chunking Function
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks


def vectorize_and_save(text_data, index_path,  chunk_data_path="chunks.txt"):
    chunks = chunk_text(text_data)

    # Save chunks to a file
    with open(chunk_data_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    vectors = embed(chunks)  # Generate embeddings
    vectors = vectors.numpy()  # Convert to NumPy for FAISS

    # Initialize FAISS index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 similarity

    # Add vectors to FAISS index
    index.add(vectors)
    faiss.write_index(index, index_path)
    return len(chunks)


if __name__ == "__main__":
    index_path = "faiss_index.index"
    input_type = input("Enter input type (pdf/website): ").strip().lower()

    if input_type == "pdf":
        pdf_path = input("Enter PDF file path: ").strip()
        text_data = extract_text_from_pdf(pdf_path)
    elif input_type == "website":
        url = input("Enter website URL: ").strip()
        text_data = extract_text_from_website(url)
    else:
        print("Invalid input type.")
        exit()

    chunks_count = vectorize_and_save(text_data, index_path)
    print(f"Successfully vectorized {chunks_count} chunks and saved to {index_path}.")
