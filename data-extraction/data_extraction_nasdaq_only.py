import gzip
import json
import glob
import os
import time

from bs4 import BeautifulSoup
import spacy
import string
from spacy.lang.en import English
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

from datetime import datetime
import re

# Start timer
start_time = time.time()

# Path to the sample-data directory
data_dir = "../../data/Press-Releases"
#data_dir = "dai-project/sample-data"
chunk = int(os.environ["CHUNK"])
files_per_chunk = 5

# Get sorted list of files (important, sonst Chaos!)
all_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl.gz")))

# Compute start and end indices for this chunk
start_idx = (chunk - 1) * files_per_chunk
end_idx = start_idx + files_per_chunk

# Slice files for this chunk
chunk_files = all_files[start_idx:end_idx]

print(f"CHUNK={chunk}, processing files {start_idx+1} to {end_idx}:")
print("\n".join(chunk_files))

json_list = []
for file_path in chunk_files:
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            json_list.append(json.loads(line))

# End timer (nachher)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

# Now json_list contains all the JSON objects from the .jsonl.gz files
print(f"Loaded {len(json_list)} JSON objects from .jsonl.gz files.")


#!python -m spacy download en_core_web_sm

spacy_nlp = spacy.load('en_core_web_sm')

# Load the FinBERT model and tokenizer inside the function
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_nlp = pipeline("sentiment-analysis",
                         model=model,
                         tokenizer=tokenizer,
                         truncation=True,
                         max_length=512)

def get_sentiment_scores(sentiment_scores):
    flag_sum = 0
    num_sentences = len(sentiment_scores)
    num_polarized = 0

    for s in sentiment_scores:
        if s['label'] == 'Positive':
            flag_sum += s['score']
            num_polarized += 1
        elif s['label'] == 'Negative':
            flag_sum -= s['score']
            num_polarized += 1
        # Neutral is ignored for polarized count

    score_including_neutrals = flag_sum / num_sentences if num_sentences else 0
    score_polarized_only = flag_sum / num_polarized if num_polarized else 0

    return score_including_neutrals, score_polarized_only


# Function to extract data from a single press release
def extract_data(press_release, relevant_tickers):
    # Extract date and format it as 'YYYY-MM-DD'
    date = press_release["Date"]
    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
    date = date.strftime('%Y-%m-%d')
    #print(date) # Uncomment for debugging


    # Get the HTML content of the document
    html_str = press_release["Document"]["Content"]
    text = ""
    # If HTML content exists, parse it and extract text
    if html_str:
        soup = BeautifulSoup(html_str, 'html.parser')
        # Find the body content tag and extract its text
        body_tag = soup.find('nitf:body.content')
        if body_tag:
            text = body_tag.get_text()


    # Clean the extracted text and identify tickers
    if text:
        # Function to clean patterns in the text
        def pattern_clean(text):
            import re
            # Replace bullet points with periods
            text = re.sub(r'•\s+', '. ', text)
            text = re.sub(r'•', '. ', text)
            # Remove patterns like ".#" or ".# " where # is a number
            text = re.sub(r'\.\d+\s+', '. ', text)
            text = re.sub(r'\.\d+', '. ', text)
            return text
        text = pattern_clean(text)

        # Use spaCy to split the text into sentences
        doc = spacy_nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Function to check if a sentence is relevant (not just metadata)
        def is_relevant_sentence(sentence):
            # Regex patterns for identifying metadata sentences (dates, times, etc.)
            metadata_patterns = [
                r"\bEastern Time\b",
                r"\b\d{4}\b",                         # Year
                r"\bJanuary|\bFebruary|\bMarch|\bApril|\bMay|\bJune|\bJuly|\bAugust|\bSeptember|\bOctober|\bNovember|\bDecember\b",
                r"\b\d{1,2}:\d{2}\s*(AM|PM)?\b",      # Time
            ]
            # Check if any metadata pattern exists and if the sentence is short
            for pattern in metadata_patterns:
                if re.search(pattern, sentence) and len(sentence.split()) < 10:
                    #print(sentence) # Uncomment for debugging
                    return False
            # Optionally: filter very short sentences
            if len(sentence.split()) < 4:
                return False
            return True

        # Regex pattern to find tickers within parentheses or brackets
        pattern = r'[\(\[]([A-Za-z ]+): ?([A-Za-z0-9\.\-]+) *[\)\]]'
        matches = re.findall(pattern, text, re.IGNORECASE)
        run_analysis = False
        for m in matches:
            #print(m) # Uncomment for debugging
            # check if m[0] is nasdaq and if m[1] i relevant_tickers
            # check m[0] with case ignoring
            if m[0].lower() == "nasdaq" and m[1] in relevant_tickers:
                run_analysis = True
        if not run_analysis:
            return []
        
            
        # Filter out irrelevant sentences
        sentences = [s for s in sentences if is_relevant_sentence(s)]
        # Get sentiment scores for the relevant sentences using the loaded pipeline
        sentences_sentiment = sentiment_nlp(sentences)


        tickers_in_text = dict()
        # Iterate through sentences and their sentiment scores
        for item in zip(sentences, sentences_sentiment):
            sentence = item[0]
            sentiment = item[1]
            # Find matches for the ticker pattern in the sentence
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                #print(matches) # Uncomment for debugging
                #print(sentence) # Uncomment for debugging
                #print(sentiment) # Uncomment for debugging
                for m in matches:
                    # If the ticker is not already in the dictionary, add it with a sentiment score
                    if m[1] not in tickers_in_text.keys():
                        if sentiment["label"] == "Positive":
                            tickers_in_text[m[1]] = 1
                        elif sentiment["label"] == "Negative":
                            tickers_in_text[m[1]] = -1
                        else: # Neutral sentiment
                            tickers_in_text[m[1]] = 0
                    else:
                        # If the ticker exists, update its score based on the current sentence's sentiment
                        if sentiment["label"] == "Positive":
                            tickers_in_text[m[1]] += 1
                        elif sentiment["label"] == "Negative":
                            tickers_in_text[m[1]] -= 1


        #print(f"Tickers in text: {tickers_in_text}") # Uncomment for debugging


    # Calculate overall sentiment scores (diluted and pure) using the helper function
    polarity_diluted, polarity_pure = get_sentiment_scores(sentences_sentiment)
    #print(f"Polarity diluted: {polarity_diluted}") # Uncomment for debugging
    #print(f"Polarity pure: {polarity_pure}") # Uncomment for debugging

    # Return a list of dictionaries, each containing ticker, date, and sentiment scores
    return [
        {
            "ticker": ticker,
            "date": date,
            "polarity_diluted": polarity_diluted,
            "polarity_pure": polarity_pure,
            "polarity_immediate": tickers_in_text[ticker] # Sentiment score based on sentences containing the ticker
        } for ticker in tickers_in_text.keys()
    ]



# =======================
# Configuration
# =======================
BATCH_SIZE = 100  # number of objects per batch
OUTPUT_DIR = "extracted_data_nasdaq"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# read the txt for the relevant tickers
with open("sp500_symbol_list.txt", "r") as f:
    relevant_tickers = f.read().splitlines()

# =======================
# Timer setup
# =======================
start_time = time.time()

# =======================
# Processing
# =======================
batch = []
batch_num = 0

for idx, press_release in enumerate(json_list, start=0):
    try:
        # Extract data
        extracted = extract_data(press_release, relevant_tickers)
        for item in extracted:
            batch.append(item)
    
        # Save batch if full
        if len(batch) >= BATCH_SIZE:
            batch_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk}_batch_{batch_num}.jsonl")
            with open(batch_file, "w") as f:
                for obj in batch:
                    f.write(json.dumps(obj) + "\n")
                    f.flush()
            
            elapsed = time.time() - start_time
            print(f"💾 Saved batch {batch_num} ({len(batch)} items) — elapsed time: {elapsed:.2f} sec", flush=True)
            
            batch = []
            batch_num += 1

        if idx % 100 == 0:
            print(f"Iteration: {idx}, Batch Size {len(batch)}")

    except Exception as e:
        print(f"⚠️ Error processing index {idx}: {e}", flush=True)
        continue

# Save any remaining objects in the last batch
if batch:
    batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num}.jsonl")
    with open(batch_file, "w") as f:
        for obj in batch:
            f.write(json.dumps(obj) + "\n")
    elapsed = time.time() - start_time
    print(f"💾 Saved final batch {batch_num} ({len(batch)} items) — elapsed time: {elapsed:.2f} sec")

print("✅ Extraction completed.")
