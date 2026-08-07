# Question 3 - Sentiment Classification Pipeline

Install the dependency and run:

```powershell
pip install -r requirements.txt
python sentiment_pipeline.py
```

On its first run, the program downloads `dataset.txt` from the URL in the question. It then performs the requested preprocessing, creates TF-IDF features, trains `LogisticRegression` with an 80/20 stratified split, prints accuracy plus the confusion matrix, and creates `metrics.json`.
