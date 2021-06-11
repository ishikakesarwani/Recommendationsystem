import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title(index):
	return df[df.index == index]["title"].values[0]

def get_index(title):
	return df[df.title == title]["index"].values[0]

df=pd.read_csv("movie_dataset.csv") #df=dataframe
#print(df.columns)

dataset_points = ['keywords','cast','genres','director']

for dataset_points in dataset_points:
	df[dataset_points] = df[dataset_points].fillna('a')
def combine_dataset_points(row): 
	try:
		return row["keywords"] +" "+row["cast"] +" "+row["genres"] +" "+row["director"] 
	except:
		print ("Error:",row)

df["combined_features"] = df.apply(combine_dataset_points,axis=1)
#print("Combined features:",df["combined_features"].head())

cv=CountVectorizer()
c_matrix=cv.fit_transform(df["combined_features"])

cosine = cosine_similarity(c_matrix)
mov_likes = "Superman"

mov_index = get_index(mov_likes)
similar_movies = list(enumerate(cosine[mov_index]))

sort_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
print("Top 5 similar movies to "+mov_likes+" are:\n")
for element in sort_similar_movies:
	print(get_title(element[0]))
	i=i+1;
	if i>5:
		break
