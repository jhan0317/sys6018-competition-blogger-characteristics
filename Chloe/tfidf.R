library(XML)
library(NLP)
library(tm)

# Import raw data
raw_train = read.csv('train.csv')
raw_test = read.csv('test.csv')

# Drop the response variable and combine the training and testing datasets together
raw_all = rbind(raw_train[,1:7], raw_test) 

# Since we only care about content, we only keep the user.id and text columns
document.data.frame = as.data.frame(raw_all[,c("user.id", "text")], stringsAsFactors = FALSE)

# Aggregate the contents with the same user.id
unique_data = aggregate(document.data.frame$text,document.data.frame['user.id'],paste,collapse=' ') 
names(unique_data) = c("doc_id", "text")

blogs = VCorpus(DataframeSource(unique_data))

blogs.clean = tm_map(blogs, stripWhitespace)                          # remove extra whitespace
blogs.clean = tm_map(blogs.clean, removeNumbers)                      # remove numbers
blogs.clean = tm_map(blogs.clean, removePunctuation)                  # remove punctuation
blogs.clean = tm_map(blogs.clean, content_transformer(tolower))       # ignore case

# Fails to remove the stopwords, it takes too long to run.
# blogs.clean = tm_map(blogs.clean, removeWords, stopwords("english")) 

blogs.clean.2 = tm_map(blogs.clean, stemDocument)                     # stem all words

# recompute TF-IDF matrix using the cleaned corpus
blogs.clean.tfidf = DocumentTermMatrix(blogs.clean.2, control = list(weighting = weightTfIdf))
as.matrix(blogs.clean.tfidf[1:5,1:5])

# remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(blogs.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
as.matrix(tfidf.99[1:5,1:5])
dim(tfidf.99)

tfidf.80 = removeSparseTerms(blogs.clean.tfidf, 0.80)  # remove terms that are absent from at least 70% of documents
as.matrix(tfidf.80[1:5, 1:5])
dim(tfidf.80)

# Export the terms to csv file
df_tfidf.80 = as.matrix(tfidf.80)
write.csv(df_tfidf.80, file = "tfidf80.csv")
