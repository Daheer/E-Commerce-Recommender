from utils import idx_to_desc, search_similar_items
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"Status": "Working"}

@app.get("/bought/{idx}")
def recommend(idx: int):
    item = idx_to_desc(idx)
    recomm_clip = idx_to_desc(search_similar_items(idx, 'clip'))
    recomm_dot = idx_to_desc(search_similar_items(idx, 'dot_nn'))
    return {"Bought Item": item, "Similar Items (CLIP)": recomm_clip, "Similar Items (DOT NN)": recomm_dot}

