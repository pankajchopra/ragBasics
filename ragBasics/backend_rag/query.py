import os
import ast
import faiss
import tensorflow_hub as hub
import google.generativeai as genai
import json


def load_env():
    with open('config.json') as f:
        config = json.load(f)
    os.environ['GEMINI_API_KEY'] = config['GEMINI_API_KEY']
    os.environ['MODEL'] = config['MODEL']


# Initialize Gemini Pro Client
def initialize_gemini(api_key):
    # Replace with your Gemini Pro API key
    genai.configure(api_key=api_key)  # or genai.configure(api_key)
    return genai.GenerativeModel(model_name=os.environ['MODEL'])


# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Function to load FAISS index
def load_faiss_index(index_file_path):
    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Index file '{index_file_path}' not found!")
    return faiss.read_index(index_file_path)


# Perform a search query
def search_faiss(query, index_path, top_k=5):
    index = load_faiss_index(index_path)

    # Vectorize the query
    query_vector = embed([query]).numpy()

    # Search the FAISS index
    distances, indices = index.search(query_vector, top_k)

    return distances[0], indices[0]


def run_query_simulation(index_path):
    query = input("Enter your query: ").strip()

    # Search the index
    print("Searching for relevant results...")
    distances, indices = search_faiss(query, index_path)

    print("\nTop Results:")
    for rank, (dist, idx) in enumerate(zip(distances, indices)):
        print(f"{rank + 1}. Chunk ID: {idx}, Distance: {dist}")

#
# if __name__ == "__main__":
#     index_path = "faiss_index.index"
#
#     # Ensure the index exists
#     if not os.path.exists(index_path):
#         print(f"Index file '{index_path}' not found. Run the vectorization script first.")
#         exit()
#
#     run_query_simulation(index_path)


# Retrieve chunks from saved text data (mock implementation)
def get_chunk_by_index(_indexPath, indices, textDataPath="chunks.txt"):
    # Simulate chunk retrieval from a text file
    if not os.path.exists(textDataPath):
        raise FileNotFoundError(f"Text data file '{textDataPath}' not found!")

    # Load all chunks (one per line)
    with open(textDataPath, 'r') as file:
        chunks = file.readlines()

    # Return selected chunks
    selected_chunks = [chunks[idx].strip() for idx in indices if idx < len(chunks)]
    return selected_chunks


def perform_rag_with_gemini(query, context, _geminiModel):
    # Prepare the payload for Gemini Pro
    payload = {
                "text": query + "\n" + "\n".join(context)

        # "systemInstruction": {
        #     "role": "system",
        #     "parts": [
        #                 {
        #                     "text": "Answer as concisely as possible in less than 256 tokens."
        #                  },
        #                 {
        #                     "text": "\n".join(context)
        #                 }
        #             ]
        # },
        # "tools": [{}],
        # "generationConfig": {
        #     "temperature": 0.7,
        #     "maxOutputTokens": 256,
        #     "topP": 0.8,
        #     "topK": 40,
        #     "stopSequences": [],
        # },
        # "labels": {
        #     "type": "rag",
        #     "filetype": "pdf"
        # }
    }
    if is_valid_dict_string(payload):
        # Generate response
        # response = _geminiModel.generate_content(query+"\n"+"\n".join(context))
        response = _geminiModel.generate_content(payload)
        return response.text
    else:
        print("Error: Context is not a dictionary.")

    # response = _geminiModel.generate_content(query+"\n"+"\n".join(context))
    return "Error: Context is not a dictionary."


def is_valid_dict_string(string):
    try:

        # Check if the result is a dictionary
        return isinstance(string, dict)
    except (ValueError, SyntaxError):
        # If evaluation fails, the string is not a valid dictionary
        return False


def run_query_simulation_with_gemini(indexPath, textData_path, geminiModel):
    query = input("Enter your query: ").strip()

    # Search the FAISS index
    print("Searching for relevant results...")
    distances, indices = search_faiss(query, indexPath)

    # Retrieve top chunks
    context = get_chunk_by_index(index_path, indices, textData_path)

    print("\nRetrieved Context:")
    for idx, chunk in enumerate(context, start=1):
        print(f"{idx}. {chunk}")

    # Perform RAG with Gemini Pro
    print("\nGenerating response with Gemini Pro...")
    gemini_response = perform_rag_with_gemini(query, context, geminiModel)

    print("\nGemini Pro Response:")
    print(gemini_response)


if __name__ == "__main__":
    load_env()
    index_path = "faiss_index.index"
    text_data_path = "chunks.txt"  # File containing the indexed text chunks

    # Ensure the index and data file exist
    if not os.path.exists(index_path) or not os.path.exists(text_data_path):
        print("Index or text data file not found. Ensure the vectorization step is complete.")
        exit()

    model = initialize_gemini(os.environ['GEMINI_API_KEY'])

    run_query_simulation_with_gemini(index_path, "chunks.txt", model)