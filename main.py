import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import random
import numpy as np

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def count_ngrams(ngrams):
    ngram_counts = defaultdict(int)
    for ngram in ngrams:
        ngram_counts[ngram] += 1
    return ngram_counts

def calculate_probabilities(ngram_counts):
    total_ngrams = sum(ngram_counts.values())
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    return probabilities

def plot_bigram_frequencies(ngram_counts):
    bigrams = list(ngram_counts.keys())
    counts = list(ngram_counts.values())
    plt.figure(figsize=(10, 8))
    plt.barh(bigrams, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.title('Bigram Frequencies')
    plt.gca().invert_yaxis()
    plt.show()

def top_n_bigrams(ngram_counts, n=10):
    return dict(sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)[:n])

def generate_random_bigram(ngram_counts):
    bigrams = list(ngram_counts.keys())
    return random.choice(bigrams)

def generate_text(ngram_counts, start_word, length=20):
    current_word = start_word
    text = [current_word]
    for _ in range(length - 1):
        possible_next = [ngram for ngram in ngram_counts if ngram.startswith(current_word)]
        if not possible_next:
            break
        next_ngram = random.choices(possible_next, weights=[ngram_counts[ngram] for ngram in possible_next])[0]
        next_word = next_ngram.split()[1]
        text.append(next_word)
        current_word = next_word
    return " ".join(text)

def plot_word_cloud(ngram_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('N-gram Word Cloud')
    plt.show()

def calculate_perplexity(ngram_counts, tokens):
    n = len(next(iter(ngram_counts)).split())
    N = len(tokens)
    log_prob = 0
    for i in range(N - n + 1):
        ngram = " ".join(tokens[i:i+n])
        if ngram in ngram_counts:
            log_prob += np.log2(ngram_counts[ngram])
        else:
            log_prob += np.log2(1 / (sum(ngram_counts.values()) + len(ngram_counts)))
    return 2 ** (-log_prob / N)

def plot_ngram_heatmap(ngram_counts, n=2):
    if n != 2:
        raise ValueError("This function only supports bigrams (n=2)")
    
    bigrams = [ngram.split() for ngram in ngram_counts.keys()]
    words = list(set([word for bigram in bigrams for word in bigram]))
    
    heatmap_data = np.zeros((len(words), len(words)))
    word_to_index = {word: idx for idx, word in enumerate(words)}
    
    for bigram, count in ngram_counts.items():
        first, second = bigram.split()
        heatmap_data[word_to_index[first]][word_to_index[second]] = count
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, xticklabels=words, yticklabels=words, cmap="YlGnBu", annot=True, fmt="g")
    plt.title("Bigram Heatmap")
    plt.xlabel("Next Word")
    plt.ylabel("Previous Word")
    plt.show()

# Sample text
text = "I love natural language processing and I love machine learning. This is an example text to test the bigram probabilities."

# Preprocess text
tokens = preprocess(text)

# Generate bigrams
bigrams = generate_ngrams(tokens, 2)

# Count bigrams
bigram_counts = count_ngrams(bigrams)

# Calculate probabilities
bigram_probabilities = calculate_probabilities(bigram_counts)

# Print bigram probabilities
print("Bigram Probabilities:")
for bigram, prob in bigram_probabilities.items():
    print(f"{bigram}: {prob:.4f}")

# Plot bigram frequencies
plot_bigram_frequencies(bigram_counts)

# Print top 5 bigrams
top_bigrams = top_n_bigrams(bigram_counts, 5)
print("\nTop 5 Bigrams:")
for bigram, count in top_bigrams.items():
    print(f"{bigram}: {count}")

# Generate a random bigram
random_bigram = generate_random_bigram(bigram_counts)
print(f"\nRandom Bigram: {random_bigram}")

# Generate text
start_word = "i"
generated_text = generate_text(bigram_counts, start_word)
print(f"\nGenerated Text: {generated_text}")

# Plot word cloud
plot_word_cloud(bigram_counts)

# Calculate perplexity
perplexity = calculate_perplexity(bigram_counts, tokens)
print(f"\nPerplexity: {perplexity:.4f}")

# Plot ngram heatmap
plot_ngram_heatmap(bigram_counts)
