import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load the current master database
file_path = ROOT / 'data' / 'processed' / 'io_construct_library_master_database.csv'
df = pd.read_csv(file_path)

# 1. Keywords to REMOVE (only from OpenAlex sources)
# These are the "leaked" terms that are clearly not psychological constructs
trash_keywords = [
    'chemistry', 'polymer', 'crystallography', 'enzyme', 'biochemistry', 
    'electrochemistry', 'molecule', 'physics', 'geology', 'astronomy', 
    'biological', 'metabolism', 'dna', 'rna', 'protein', 'alloy', 'catalysis'
]

def should_keep(row):
    # Always keep O*NET constructs (they are pre-filtered and high value)
    if row['Source'] == 'O*NET':
        return True
    
    # Check OpenAlex for trash keywords in name or definition
    name = str(row['Construct_Name']).lower()
    desc = str(row['Definition_Text']).lower()
    
    # If it's a "leaked" science term, drop it
    for word in trash_keywords:
        if word in name or word in desc:
            return False
            
    # Filter out very low paper counts (unless they have a DOI)
    # This removes the "1-off" noise Patrick mentioned
    if row['Paper_Count'] < 3 and pd.isna(row['Reference_URLs']):
        return False
        
    return True

# Apply filtering
initial_count = len(df)
df_clean = df[df.apply(should_keep, axis=1)]
final_count = len(df_clean)

# Save the cleaned version
output_path = ROOT / 'data' / 'processed' / 'cleaned_master_database.csv'
df_clean.to_csv(output_path, index=False)

print(f"Scrub Complete!")
print(f"Initial Rows: {initial_count}")
print(f"Removed: {initial_count - final_count}")
print(f"Cleaned Rows: {final_count}")
print(f"Saved to: {output_path}")
