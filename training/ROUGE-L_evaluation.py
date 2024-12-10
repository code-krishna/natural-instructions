print("HI")

from rouge_score import rouge_scorer
import numpy as np

# Define the generated responses and ground truth list
generated_responses = [
    "The cat sits on the mat.",
    "The dog barks at the mailman.",
    "Birds fly high in the sky."
]

ground_truths = [
    "A cat is on the mat.",
    "A dog is barking at the postman.",
    "Birds soar in the sky."
]

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Initialize lists to hold scores
rouge_L_scores = []

# Calculate ROUGE-L for each pair of generated response and ground truth
for gen, ref in zip(generated_responses, ground_truths):
    scores = scorer.score(ref, gen)
    rouge_L_scores.append(scores['rougeL'].fmeasure)  # We care about the F1 score here

# Compute average ROUGE-L F1 score across all responses
average_rouge_L_f1 = np.mean(rouge_L_scores)

# Print the results
print(f"ROUGE-L F1 Scores for each pair: {rouge_L_scores}")
print(f"Average ROUGE-L F1 Score: {average_rouge_L_f1:.4f}")
