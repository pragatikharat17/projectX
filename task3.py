import math
from collections import Counter

def bleu_score(references, candidate):
    # Function to calculate n-gram precision
    def ngram_precision(candidate_ngrams, reference_ngrams):
        clipped_counts = Counter(candidate_ngrams) & Counter(reference_ngrams)
        return sum(clipped_counts.values()) / max(1, len(candidate_ngrams))
    
    # Function to calculate brevity penalty
    def brevity_penalty(candidate_length, reference_lengths):
        closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
        if candidate_length > closest_ref_length:
            return 1
        else:
            return math.exp(1 - closest_ref_length / candidate_length)
    
    # Tokenize sentences into n-grams
    references_ngrams = [tuple(ref.split()) for ref in references]
    candidate_ngrams = tuple(candidate.split())
    
    # Calculate precision for each n-gram order
    precisions = [ngram_precision(candidate_ngrams[i:], references_ngrams) for i in range(4)]
    
    # Calculate brevity penalty
    candidate_length = len(candidate_ngrams)
    reference_lengths = [len(ref_ngram) for ref_ngram in references_ngrams]
    penalty = brevity_penalty(candidate_length, reference_lengths)
    
    # Calculate BLEU score
    bleu = penalty * math.exp(sum(math.log(precision + 1e-10) for precision in precisions) / 4)
    
    return bleu

# Test cases

# test case 1
references_1 = [
    "It is a guide to action that ensures that the military will forever heed Party commands",
    "It is the guiding principle which guarantees the military forces always being under the command of the Party",
    "It is the practical guide for the army always to heed the directions of the party"
]
candidate_1 = "It is a guide to action which ensures that the military always obeys the commands of the party"

# test case 2
references_2 = [
    "It is a guide to action that ensures that the military will forever heed Party commands",
    "It is the guiding principle which guarantees the military forces always being under the command of the Party",
    "It is the practical guide for the army always to heed the directions of the party"
]
candidate_2 = "It is the to action the troops forever hearing the activity guidebook that party direct"

# test case 3
references_3 = [
    "The cat is on the mat",
    "The cat sits on the mat"
]
candidate_3 = "A cat is on the mat"

# test case 4
references_4 = [
    "The cat is on the mat",
    "The cat sits on the mat"
]
candidate_4 = "The cat sits on the mat"

# Calculate BLEU scores for all test cases
score_1 = bleu_score(references_1, candidate_1)
score_2 = bleu_score(references_2, candidate_2)
score_3 = bleu_score(references_3, candidate_3)
score_4 = bleu_score(references_4, candidate_4)

# Print BLEU scores
print("BLEU Score Test Case 1:", score_1)
print("BLEU Score Test Case 2:", score_2)
print("BLEU Score Test Case 3:", score_3)
print("BLEU Score Test Case 4:", score_4)
