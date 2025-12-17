Semantic Book Recommender with LLMs  

Live demo: https://huggingface.co/spaces/Shiwana/book-recommender  

This project is an end-to-end semantic book recommendation system built with large language models (LLMs), vector search, and a Gradio web application. It enables users to search for books in natural language and returns the most relevant ones based on meaning rather than simple keyword matching.

### Project Structure

- `data-exploration.ipynb`  
  Performs exploratory data analysis on the raw book dataset, including correlation checks, missing-value inspection, and feature engineering. Irrelevant or incomplete columns are removed, useful new features are created, and the cleaned dataset is exported as `books_cleaned.csv`.

- `vector-search.ipynb`  
  Builds the vector database for semantic search by generating embeddings for book descriptions. This notebook powers natural language queries so users can retrieve books that are most similar in meaning to their input.

- `text-classification.ipynb`  
  Uses zero-shot text classification with LLMs on the cleaned dataset to label each book as “fiction” or “non-fiction”. The resulting categories are stored in `books_with_categories.csv`, allowing users to filter recommendations by type.

- `sentiment-analysis.ipynb`  
  Applies LLM-based sentiment analysis to book descriptions to derive scores for seven emotions. These scores are saved in `books_with_emotions.csv` and can be used to sort or filter recommendations by emotional tone.

- `gradio-dashboard.py`  
  Implements the main Gradio web interface that connects the vector search, classifications, and sentiment data. This dashboard lets users enter queries, apply filters, and explore recommended books through an intuitive UI.

  requirements.txt – All dependencies required to run the notebooks and the app.
