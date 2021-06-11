from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
text = ["Lucknow Dehradun Lucknow","Dehradun Dehradun Lucknow"]  

#find count of each word

cv=CountVectorizer() 


c_matrix=cv.fit_transform(text)
print (c_matrix.toarray()) 


#finding the cosine similarity

similar_scoring=cosine_similarity(c_matrix)
print (similar_scoring)