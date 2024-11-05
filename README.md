# TripAdvisor Hotel Review Analysis
This script scrapes hotel reviews from TripAdvisor, analyzes them, and builds a predictive model to estimate the hotel's rating.
## Features
- **Data Scraping:** Retrieves hotel reviews from TripAdvisor using `requests` and `BeautifulSoup`.
- **Data Preprocessing:** Cleans the text reviews by:
    - Converting to lowercase.
    - Removing non-alphabetic characters.
    - Removing stop words.
- **Text Representation:** Uses `TfidfVectorizer` to transform text reviews into a numerical vector representation.
- **Exploratory Data Analysis:**
    - Generates a distribution plot of review ratings.
    - Creates word clouds to visualize frequent words in positive and negative reviews.
- **Machine Learning Models:** Trains several machine learning models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Multilayer Perceptron (MLP)
- **Model Evaluation:** Evaluates the accuracy of each model on a test set.
- **Hyperparameter Optimization:** Uses `GridSearchCV` to find optimal parameters for the Random Forest model.
- **Cross-Validation:** Performs cross-validation to assess the robustness of the models.
- **Confusion Matrices:** Generates confusion matrices for each model to visualize their classification performance.
## Installation
1. Install the necessary packages:
   ```bash
   pip install -r requirements (2).txt
