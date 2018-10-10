library(XML)
library(NLP)
library(tm)

# Import raw data
raw_train = read.csv('train.csv')
raw_test = read.csv('test.csv')
x_train = raw_train[,1:7]
x_test = raw_test

# Since we only care about content, we only keep the user.id and text columns
document.data.frame.train = as.data.frame(x_train[,c("user.id", "text")], stringsAsFactors = FALSE)
document.data.frame.test = as.data.frame(x_test[,c("user.id", "text")], stringsAsFactors = FALSE)

# Aggregate the contents with the same user.id
unique.data.train = aggregate(document.data.frame.train$text,document.data.frame.train['user.id'],paste,collapse=' ') 
unique.data.test = aggregate(document.data.frame.test$text,document.data.frame.test['user.id'],paste,collapse=' ') 

names(unique.data.train) = c("doc_id", "text")
names(unique.data.test) = c("doc_id", "text")

blogs.train = VCorpus(DataframeSource(unique.data.train))
blogs.test = VCorpus(DataframeSource(unique.data.test))

blogs.clean.train = tm_map(blogs.train, stripWhitespace)                        # remove extra whitespace
blogs.clean.train = tm_map(blogs.clean.train, removeNumbers)                    # remove numbers
blogs.clean.train = tm_map(blogs.clean.train, removePunctuation)                # remove punctuation
blogs.clean.train = tm_map(blogs.clean.train, content_transformer(tolower))     # ignore case
# Fails to remove the stopwords, it takes too long to run.
# blogs.clean.train = tm_map(blogs.clean.train, removeWords, stopwords("english")) 
blogs.clean.train = tm_map(blogs.clean.train, stemDocument)                     # stem all words

blogs.clean.test = tm_map(blogs.test, stripWhitespace)                          # remove extra whitespace
blogs.clean.test = tm_map(blogs.clean.test, removeNumbers)                      # remove numbers
blogs.clean.test = tm_map(blogs.clean.test, removePunctuation)                  # remove punctuation
blogs.clean.test = tm_map(blogs.clean.test, content_transformer(tolower))       # ignore case
# Fails to remove the stopwords, it takes too long to run.
# blogs.clean.test = tm_map(blogs.clean.test, removeWords, stopwords("english")) 
blogs.clean.test = tm_map(blogs.clean.test, stemDocument)                       # stem all words

# recompute TF-IDF matrix using the cleaned corpus
blogs.clean.tfidf.train = DocumentTermMatrix(blogs.clean.train, control = list(weighting = weightTfIdf))
blogs.clean.tfidf.test = DocumentTermMatrix(blogs.clean.test, control = list(weighting = weightTfIdf))

# remove sparse terms at various thresholds.
# First, we removed the terms that are absent from 99% of the documents.
tfidf.99.train = removeSparseTerms(blogs.clean.tfidf.train, 0.99)  
as.matrix(tfidf.99.train[1:5,1:5])
# Terms
# Docs “hey “how         “i’m “if         “it
#   1    0    0 0.0000000000   0 0.000000000
#   3    0    0 0.0000000000   0 0.000000000
#   4    0    0 0.0000000000   0 0.000000000
#   5    0    0 0.0009035494   0 0.000914463
#   6    0    0 0.0001436164   0 0.000000000
dim(tfidf.99.train)
# [1] 12880 10881

# There are still too many terms, therefore we remove terms that are absent from 80% of the documents.
tfidf.80.train = removeSparseTerms(blogs.clean.tfidf.train, 0.80)  
as.matrix(tfidf.80.train[1:5, 1:5])
# Terms
# Docs          abl        about         abov      absolut       accept
#    1 0.0000000000 0.0004583561 0.0000000000 0.0000000000 0.0000000000
#    3 0.0008859245 0.0004583561 0.0000000000 0.0000000000 0.0000000000
#    4 0.0000000000 0.0003716492 0.0000000000 0.0000000000 0.0000000000
#    5 0.0005858814 0.0003599561 0.0002753518 0.0002778795 0.0013301960
#    6 0.0002560911 0.0006609714 0.0002188316 0.0006625214 0.0001902875
dim(tfidf.80.train)
# [1] 12880  1072

tfidf.80.test = removeSparseTerms(blogs.clean.tfidf.test, 0.80)  
as.matrix(tfidf.80.test[1:5, 1:5])
# Terms
# Docs          abl        about         abov      absolut       accept
#   2  0.0000000000 0.0005086734 0.0000000000 0.0000000000 0.0000000000
#   8  0.0003948863 0.0003004314 0.0000000000 0.0000000000 0.0007039919
#   9  0.0002712702 0.0007064673 0.0001170513 0.0004798181 0.0005394141
#   10 0.0015568233 0.0004342942 0.0011643821 0.0005966312 0.0000000000
#   11 0.0013949666 0.0005306487 0.0000000000 0.0000000000 0.0000000000
dim(tfidf.80.test)
# [1] 6440 1041

# Export the terms to csv file
df.tfidf.80.train = as.matrix(tfidf.80.train)
df.tfidf.80.test = as.matrix(tfidf.80.test)
write.csv(df.tfidf.80.train, file = "tfidf80_train.csv")  # The csv file is too large to be uploaded to gitHub
write.csv(df.tfidf.80.test, file = "tfidf80_test.csv")  # The csv file is too large to be uploaded to gitHub
