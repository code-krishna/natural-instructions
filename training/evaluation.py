import re
from difflib import SequenceMatcher
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import pipeline
from rouge_score import rouge_scorer
import numpy as np
import json

# Load your JSON data
with open("output_reponses_trained_processed.json", "r") as file:
    data = json.load(file)

# Function to preprocess responses (remove non-alphanumeric special characters)
def preprocess_response(response):
    return re.sub(r"[^a-zA-Z0-9\s]", "", response).strip()  # Keep only alphanumeric characters and spaces

# Function to compute Exact Match
def exact_match(correct, generated):
    return correct.strip() == generated.strip()

# Function to check if correct response is a substring of generated response
def is_substring(correct, generated):
    return correct.strip() in generated.strip()

# Function to compute Levenshtein Similarity
def levenshtein_similarity(correct, generated):
    return SequenceMatcher(None, correct, generated).ratio()

# Semantic Similarity using Transformers
semantic_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
def semantic_similarity(correct, generated):
    # Extract embeddings and reduce them to a single vector by averaging
    correct_emb = np.mean(semantic_model(correct)[0], axis=0)  # Average over tokens
    generated_emb = np.mean(semantic_model(generated)[0], axis=0)  # Average over tokens

    # Calculate cosine similarity
    dot_product = np.dot(correct_emb, generated_emb)
    norm_correct = np.linalg.norm(correct_emb)
    norm_generated = np.linalg.norm(generated_emb)
    cosine_sim = dot_product / (norm_correct * norm_generated)
    return cosine_sim

# ROUGE Score Calculation
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def compute_rouge(correct, generated):
    scores = rouge.score(correct, generated)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

# Evaluation Loop
results = {
    "exact_match": [],
    "substring_match": [],
    "levenshtein_similarity": [],
    "semantic_similarity": [],
    "rouge1": [],
    "rouge2": [],
    "rougeL": []
}

for item in data:
    # Preprocess the responses
    correct = preprocess_response(str(item["correct_response"]))
    generated = preprocess_response(str(item["generated_response"]))
    # print(correct, ", " ,generated)
    
    # Exact Match
    results["exact_match"].append(exact_match(correct, generated))
    
    # Substring Match
    results["substring_match"].append(is_substring(correct, generated))
    
    # Levenshtein Similarity
    results["levenshtein_similarity"].append(levenshtein_similarity(correct, generated))
    
    # Semantic Similarity
    results["semantic_similarity"].append(semantic_similarity(correct, generated))
    
    # ROUGE Scores
    rouge1, rouge2, rougeL = compute_rouge(correct, generated)
    results["rouge1"].append(rouge1)
    results["rouge2"].append(rouge2)
    results["rougeL"].append(rougeL)

# Compute Metrics
exact_match_accuracy = sum(results["exact_match"]) / len(results["exact_match"])
substring_match_accuracy = sum(results["substring_match"]) / len(results["substring_match"])
average_levenshtein = sum(results["levenshtein_similarity"]) / len(results["levenshtein_similarity"])
average_semantic_similarity = sum(results["semantic_similarity"]) / len(results["semantic_similarity"])
average_rouge1 = sum(results["rouge1"]) / len(results["rouge1"])
average_rouge2 = sum(results["rouge2"]) / len(results["rouge2"])
average_rougeL = sum(results["rougeL"]) / len(results["rougeL"])

# Print Metrics
print("Metrics:")
print(f"Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%")
print(f"Substring Match Accuracy: {substring_match_accuracy * 100:.2f}%")
print(f"Average Levenshtein Similarity: {average_levenshtein * 100:.2f}%")
print(f"Average Semantic Similarity: {average_semantic_similarity * 100:.2f}%")
print(f"Average ROUGE-1 Score: {average_rouge1 * 100:.2f}%")
print(f"Average ROUGE-2 Score: {average_rouge2 * 100:.2f}%")
print(f"Average ROUGE-L Score: {average_rougeL * 100:.2f}%")
