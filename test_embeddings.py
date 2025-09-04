from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode("Test embedding")
print(emb.shape)