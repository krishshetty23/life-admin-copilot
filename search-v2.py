from sentence_transformers import SentenceTransformer
import numpy as np

# loading the embedding model (this is the AI part)
print("Loading the AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# reading the file
with open('myProfile.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# turning each line into embeddings (words → numbers with meaning)
print("\nCreating embeddings for your profile...")
lineEmbeddings = model.encode(lines)
print(f"Created embeddings for {len(lines)} lines")

# asking a question
question = "Where did I study?"

# turning question into embedding
print(f"\nQuestion: {question}")
questionEmbeddings = model.encode(question)

# finding the most similar line using cosine similarity
similarities = np.dot(lineEmbeddings, questionEmbeddings.T).flatten()
bestMatch_idx =similarities.argmax()

print(f"Best match: {lines[bestMatch_idx]}")
print(f"Confidence: {similarities[bestMatch_idx]:2f}")