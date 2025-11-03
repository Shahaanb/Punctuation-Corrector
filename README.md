# AI Punctuation & Capitalization Restoration

This project uses a fine-tuned **T5-Small** model combined with a **rule-based post-processing pipeline** to automatically add correct punctuation and capitalization to unformatted, raw text.

It's designed to take messy, lowercase text without any punctuation and transform it into a clean, grammatically correct, and readable format.

## The Problem

It takes text input like this:

> "the quick brown fox jumps over the lazy dog this is a classic sentence used for typing practice but it also serves as a good test for our model i wonder if it will know where to put the period and how to capitalize the word 'this' in the middle of the text it's a non-trivial task"

## The Solution

And intelligently transforms it into this:

> "The quick brown fox jumps over the lazy dog. This is a classic sentence used for typing practice, but it also serves as a good test for our model. I wonder if it will know where to put the period and how to capitalize the word. This in the middle of the text, it's a nontrivial task..."

## How It Works: A Hybrid Approach

This isn't just a single model; it's a two-stage pipeline that combines the strengths of a neural network with the precision of rule-based corrections.

### Stage 1: T5-Small AI Model

A **T5-Small** (a Text-to-Text Transfer Transformer model) was fine-tuned on the `wikitext` dataset.

* **Training Task:** The model was trained to "translate" text from a "broken" format (all lowercase, all punctuation removed) to its original, correct format.
* **Prefix:** It was trained using the task prefix `correct: `, teaching it to restore grammar when it sees that prompt.

### Stage 2: Rule-Based Post-Processing

The AI's output is good, but not always perfect. A secondary Python script cleans up the model's predictions using a series of regex rules to fix common, high-frequency errors. This includes:

* **Contraction Fixing:** Corrects `its` -> `it's`, `dont` -> `don't`, etc.
* **Lone "i" Capitalization:** Fixes `i` -> `I`.
* **Spacing Cleanup:** Removes erroneous spaces before commas/periods (e.g., `word .` -> `word.`).
* **Sentence Capitalization:** Ensures any letter following a `.`, `!`, or `?` is capitalized.

## Key Features

* **Restores Punctuation:** Accurately places periods, commas, and question marks.
* **Restores Capitalization:** Correctly capitalizes the start of sentences.
* **Fixes Contractions:** Intelligently corrects common English contractions.
* **Capitalizes "I"**: Understands the context of the lone pronoun "I" and capitalizes it.
* **Long Text Support:** Includes a text-chunking function (`correct_large_paragraph`) to process long documents without exceeding the model's token limit.
* **High Accuracy:** The testing notebook demonstrates **~94% similarity** (using `fuzzywuzzy`) between the pipeline's output and the ground-truth text.

## Project Notebooks

* `Model_Training_Final.ipynb`: The complete notebook for training the T5-Small model from scratch on the `wikitext` dataset. This is where the core AI model (`.pt` file) is created and saved.
* `Model_Testing.ipynb`: A notebook that loads the fine-tuned model and provides the complete, end-to-end pipeline for inference (including text chunking and post-processing). **This is the notebook to use for running predictions.**

## How to Use (Inference)

You can easily run predictions using the `Model_Testing.ipynb` notebook.

### 1. Setup

Install the required libraries:

```bash
pip install transformers torch fuzzywuzzy
```

### 2. Load Your Trained Model

You must first have your trained model checkpoint (e.g., the `checkpoint-31074` folder saved by the training notebook).

Upload this folder to your environment (or mount your Google Drive if using Colab) and update the `MODEL_PATH` variable:

```python
# This MUST match the path to your saved checkpoint
MODEL_PATH = "/content/drive/MyDrive/final-punctuation-model/checkpoint-31074"

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Setup device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

### 3. Run the Correction Pipeline

Copy the helper functions from `Model_Testing.ipynb` into your script:

1. `correct(text)`: The core model prediction function.  
2. `correct_large_paragraph(long_text)`: The chunking function.  
3. `check(sentence)`: The wrapper that decides which function to use (chunking or single-pass).  
4. `post_process_refinement(model_output_text)`: The rule-based cleanup pipeline.

You can then run the full pipeline on any new text:

```python
# Your messy, unformatted text
my_text = "hello my name is shahaan what is yours i hope this works"

# 1. Get the raw model output
model_output = check(my_text)

# 2. Clean it with the rule-based post-processor
polished_output = post_process_refinement(model_output)

print(f"Input:    '{my_text}'")
print(f"Output:   '{polished_output}'")

# --- Example with a long paragraph ---
long_text = "the quick brown fox jumps over the lazy dog this is a classic sentence used for typing practice but it also serves as a good test for our model i wonder if it will know where to put the period and how to capitalize the word 'this' in the middle of the text it's a non-trivial task because the model has to understand context not just individual words for example will it know what to do with a sentence like this what do you think the final output will be i am very excited to see the results"

polished_long_output = post_process_refinement(check(long_text))
print(f"\n--- Long Text Example ---")
print(f"Input:    '{long_text}'")
print(f"Output:   '{polished_long_output}'")
```

**Example Output:**

```
Input:    'hello my name is shahaan what is yours i hope this works'
Output:   'Hello, my name is Shahaan, what is yours? I hope this works.'

--- Long Text Example ---
Input:    'the quick brown fox jumps over the lazy dog...'
Output:   'The quick brown fox jumps over the lazy dog. This is a classic sentence used for typing practice, but it also serves as a good test for our model. I wonder if it will know where to put the period and how to capitalize the word. This in the middle of the text, it's a nontrivial task because the model has to understand context, not just individual words. For example, Will it know what to do with a sentence like this? What do you think the final output will be? I am very excited to see the results?'
```

## How to Re-Train the Model

1. Open `Model_Training_Final.ipynb` (preferably in Google Colab with a GPU runtime).  
2. Mount your Google Drive when prompted.  
3. Define your `output_dir` (e.g., `/content/drive/MyDrive/my-new-punctuation-model`).  
4. Run all cells. The notebook will:
   * Download and preprocess the `wikitext` dataset.  
   * Load the base `t5-small` model.  
   * Fine-tune the model on the punctuation task (this will take some time).  
   * Save the best-performing checkpoint to your specified Google Drive path.  
5. Once finished, update the `MODEL_PATH` in the `Model_Testing.ipynb` to point to your new checkpoint.

## Technologies Used

* **Python 3**
* **PyTorch**
* **Hugging Face `transformers`:** For the T5 model, Tokenizer, and Trainer.
* **Hugging Face `datasets`:** For the `wikitext` dataset.
* **FuzzyWuzzy:** For similarity-scoring the model's performance.
* **Google Colab & Drive:** For GPU-powered training and model storage.

<!-- end list -->
