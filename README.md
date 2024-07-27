# N-gram Language Model

This task implements a simple N-gram language model in Python. It includes functionalities for bigram generation, frequency analysis, probability calculation, text generation, visualization, and evaluation.

## Features

- **Text Preprocessing**: Convert text to lowercase and tokenize it.
- **Bigram Generation**: Create bigrams from the tokenized text.
- **Frequency Count**: Count the occurrences of each bigram.
- **Probability Calculation**: Calculate the probabilities of each bigram.
- **Bigram Frequency Plot**: Visualize the frequency of bigrams using a bar chart.
- **Top N Bigrams**: Retrieve the top N most frequent bigrams.
- **Random Bigram Generation**: Generate random bigrams based on their frequencies.
- **Text Generation**: Generate text based on the bigram probabilities.
- **Word Cloud Visualization**: Create a word cloud from the bigram counts.
- **Perplexity Calculation**: Calculate the perplexity of a given text sequence.
- **N-gram Heatmap**: Visualize the bigram frequencies using a heatmap.

## Requirements

- Python 3.x
- matplotlib
- seaborn
- wordcloud
- numpy

You can install the required libraries using:

```bash
pip install matplotlib seaborn wordcloud numpy
```

## Usage

clone the repo and run the main.py file

```bash
python main.py
```

### Function Descriptions

- `preprocess(text)`: Converts text to lowercase and tokenizes it into words.
- `generate_ngrams(tokens, n)`: Generates N-grams from the tokenized text.
- `count_ngrams(ngrams)`: Counts the occurrences of each N-gram.
- `calculate_probabilities(ngram_counts)`: Calculates the probabilities of each N-gram.
- `plot_bigram_frequencies(ngram_counts)`: Plots a bar chart of bigram frequencies.
- `top_n_bigrams(ngram_counts, n)`: Returns the top N most frequent bigrams.
- `generate_random_bigram(ngram_counts)`: Generates a random bigram based on their frequencies.
- `generate_text(ngram_counts, start_word, length)`: Generates a sequence of text starting from a given word.
- `plot_word_cloud(ngram_counts)`: Generates a word cloud from the bigram counts.
- `calculate_perplexity(ngram_counts, tokens)`: Calculates the perplexity of a given text sequence.
- `plot_ngram_heatmap(ngram_counts, n)`: Plots a heatmap of bigram frequencies.

### Running the Code

1. **Install the required libraries**: Ensure you have the necessary Python libraries installed.
2. **Run the sample code**: Use the provided sample text or your own text to preprocess, generate bigrams, calculate probabilities, visualize, and analyze.

### Example Output

The code will output the following:
- Bigram probabilities.
- A bar chart of bigram frequencies.
- Top N most frequent bigrams.
- A randomly generated bigram.
- A generated sequence of text.
- A word cloud of the bigrams.
- The perplexity score of the model.
- A heatmap of bigram frequencies.

Feel free to customize the code to suit your needs and explore the fascinating world of N-gram language models!

