import pandas as pd
import requests
import time
from sqlalchemy import create_engine, text
from config import load_env_file, get_required_env

load_env_file()
POSTGRES_URL = get_required_env("POSTGRES_URL")
CF_TOKEN = get_required_env("CLOUDFLARE_API_TOKEN")
CF_ACCOUNT = get_required_env("CLOUDFLARE_ACCOUNT_ID")

engine = create_engine(POSTGRES_URL)

# 1. Load Cleaned Data
df = pd.read_csv('/Users/jomus/Code/construct-library/data/processed/cleaned_master_database.csv')

# 2. Setup DB Table
print("Setting up database table...")
setup_sql = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS master_constructs;
CREATE TABLE master_constructs (
    id SERIAL PRIMARY KEY,
    "Construct_Name" TEXT,
    "Source" TEXT,
    "Description" TEXT,
    "Paper_Count" INTEGER,
    "Reference_URLs" TEXT,
    embedding vector(768)
);
"""
with engine.connect() as conn:
    conn.execute(text(setup_sql))
    conn.commit()

# 3. Embedding Function (Cloudflare)
def get_embedding(text_list):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/run/@cf/baai/bge-base-en-v1.5"
    headers = {"Authorization": f"Bearer {CF_TOKEN}"}
    response = requests.post(url, headers=headers, json={"text": text_list})
    if response.status_code == 200:
        return response.json()['result']['data']
    else:
        print(f"Error embedding: {response.text}")
        return None

# 4. Batch Process and Upload
batch_size = 20
total = len(df)
print(f"Starting migration of {total} constructs...")

for i in range(0, total, batch_size):
    batch = df.iloc[i:i+batch_size]
    texts = batch['Definition_Text'].fillna(batch['Construct_Name']).tolist()
    
    embeddings = get_embedding(texts)
    if not embeddings:
        continue
        
    for j, (_, row) in enumerate(batch.iterrows()):
        insert_sql = text("""
            INSERT INTO master_constructs ("Construct_Name", "Source", "Description", "Paper_Count", "Reference_URLs", embedding)
            VALUES (:name, :source, :desc, :count, :urls, :embedding)
        """)
        
        with engine.connect() as conn:
            conn.execute(insert_sql, {
                "name": row['Construct_Name'],
                "source": row['Source'],
                "desc": row['Definition_Text'],
                "count": row['Paper_Count'],
                "urls": row['Reference_URLs'],
                "embedding": str(embeddings[j])
            })
            conn.commit()
            
    print(f"Progress: {min(i+batch_size, total)}/{total}")
    time.sleep(0.5) # Rate limit safety

print("\nMigration Complete! Your library is live in Neon with 768-dim vectors.")
