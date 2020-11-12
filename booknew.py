import pandas as pd

df=pd.read_csv("book.csv",encoding='latin-1')
print(df)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(df["Book.Title"])
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
df_index = pd.Series(df.index,index=df["Book.Title"]).drop_duplicates()



# Enter your anime and number of anime's to be recommended 
def get_book_recommendations(Name, topN):
    df_id = df_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    cosine_scores_10 = cosine_scores[0:topN + 1]
    book_idx = [i[0] for i in cosine_scores_10]
    book_scores = [i[1] for i in cosine_scores_10]
    book_similar_show = pd.DataFrame(columns=["name", "Score"])
    book_similar_show["name"] = df.loc[book_idx, "Book.Title"]
    book_similar_show["Score"] = book_scores
    book_similar_show.reset_index(inplace=True)
    book_similar_show.drop(["index"], axis=1, inplace=True)
    print(book_similar_show)
    # return (anime_similar_show)

    pass


get_book_recommendations("Clara Callan",topN=15)
