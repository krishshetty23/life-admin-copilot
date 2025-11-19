from sentence_transformers import SentenceTransformer
import numpy as np
import threading

_model = None
_model_lock = threading.Lock()

# function to load the embedding model
def load_model():
    """
    Returns a cached instance of the SentenceTransformer model.
    Only loads it the first time. Thread-safe.
    """
    global _model

    # Double-checked locking pattern for thread safety
    if _model is None:
        with _model_lock:
            # Check again inside the lock in case another thread loaded it
            if _model is None:
                print("Loading the AI model...")
                # Load with device=None (works best with multi-threading)
                _model = SentenceTransformer('all-MiniLM-L6-v2', device=None)
                _model.eval()  # Set to evaluation mode
                print("Model loaded!")

    return _model


# function to load the profile context
def load_profile():
    """
    Reads the profile file and returns a list of non-empty lines.
    """
    with open('myProfile.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    return lines


# semantic search
def semantic_search(question):
    """
    Given a question and a profile file, returns the best matching line and score.
    Returns a dict like:
    {
        "question": "...",
        "best_match": "...",
        "score": 0.95,
        "lines": [... all profile lines ...]
    }
    """

    # loading the embedding model (this is the AI part)
    model = load_model()

    # reading the file
    lines = load_profile()

    # turning each line into embeddings (words → numbers with meaning)
    # print("\nCreating embeddings for your profile...")
    lineEmbeddings = model.encode(lines)
    # print(f"Created embeddings for {len(lines)} lines")

    # turning question into embedding
    # print(f"\nQuestion: {question}")
    questionEmbeddings = model.encode(question)

    # finding the most similar line using cosine similarity
    similarities = np.dot(lineEmbeddings, questionEmbeddings.T).flatten()
    bestMatch_idx = int(np.argmax(similarities))

    return {
        "question": question,
        "best_match": lines[bestMatch_idx],
        "score": float(similarities[bestMatch_idx]),
        "lines": lines,
    }


if __name__ == "__main__":
    # asking a question
    question = "Where did I study?"

    result = semantic_search(question)

    print(f"\nQuestion: {result['question']}")
    print(f"Best match: {result['best_match']}")
    print(f"Confidence: {result['score']:.4f}")