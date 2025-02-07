from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key="AIzaSyA9UgryALUMKM1qPSRVgD_3G7o3inaAMjU")

# Load the Excel file
df = pd.read_excel("Processed_Text.xlsx", engine="openpyxl")

# Convert embeddings from string to list
df["Embedding"] = df["Embedding"].apply(lambda x: np.array(eval(x)))

# groq_api_key = "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"
groq_api_key=os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

def get_llm_result(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def process_query(query):
    # Generate embedding for the query
    query_embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )["embedding"]
    query_embedding = np.array(query_embedding)

    # Compute cosine similarity
    df["Similarity"] = df["Embedding"].apply(lambda emb: cosine_similarity([query_embedding], [emb])[0][0])

    # Get the most relevant text (sorted by similarity)
    top_match = df.sort_values(by="Similarity", ascending=False).iloc[0]
    relevant_text = top_match["Chunk"]

    prompt = f"""
    Using the extracted information below:
    
    {relevant_text}
    
    Provide a brief and precise answer (3-4 lines) to the following query:
    
    {query}
    """
    return get_llm_result(prompt)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        query = request.form['query']
        result = process_query(query)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


