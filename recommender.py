import pandas as pd
import re
import nltk
# nltk.data.path.append('./nltk_data')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Loading pivot table, model, or predictions (e.g., from a pickle file)

user_final_rating = pd.read_pickle("pickle_files/user_final_rating1.pkl")
tfidf_vectorizer = pd.read_pickle("pickle_files/Tfidf_vectorizer.pkl")
model = pd.read_pickle("pickle_files/XGBoost_classifier_grid.pkl")
product_df = pd.read_csv('sample30.csv',sep=",")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(no_stop)
    return ' '.join(lemmatized)

#predicting the sentiment of the product review comments
def model_predict(text):
    tfidf_vector = tfidf_vectorizer.transform(text)
    output = model.predict(tfidf_vector)
    return output

def recommend_items(user_id, top_n=5):
    print(f"Recommending items for user: {user_id}")
    if user_id not in user_final_rating.index:
        return None
    product_list = pd.DataFrame(user_final_rating.loc[user_id].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name','reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: preprocess_text(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    print(f"Predicted sentiments for user {user_id}: {output_df['predicted_sentiment'].value_counts()}")
    return output_df

def top5_products(df):
    print("Calculating top 5 products based on positive sentiment...")
    total_product=df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name','predicted_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['reviews_text'],on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] ==  1][:5])
    print(f"Top 5 products based on positive sentiment: {output_products['name'].tolist()}")
    return output_products