library(XML)
library(NLP)
library(tm)

# Import raw data
raw_train = read.csv('x_train_for_tm.csv')
raw_test = read.csv('x_test_for_tm.csv')

# Since we only care about content, we only keep the user.id and text columns
document.data.frame.train = as.data.frame(raw_train, stringsAsFactors = FALSE)
document.data.frame.test = as.data.frame(raw_test, stringsAsFactors = FALSE)

names(document.data.frame.train) = c("doc_id", "text")
names(document.data.frame.test) = c("doc_id", "text")

blogs.train = VCorpus(DataframeSource(document.data.frame.train))
blogs.test = VCorpus(DataframeSource(document.data.frame.test))

blogs.clean.train = tm_map(blogs.train, stripWhitespace)                        # remove extra whitespace
blogs.clean.train = tm_map(blogs.clean.train, removeNumbers)                    # remove numbers
blogs.clean.train = tm_map(blogs.clean.train, stemDocument)                     # stem all words

blogs.clean.test = tm_map(blogs.test, stripWhitespace)                          # remove extra whitespace
blogs.clean.test = tm_map(blogs.clean.test, removeNumbers)                      # remove numbers
blogs.clean.test = tm_map(blogs.clean.test, stemDocument)                       # stem all words

# recompute TF-IDF matrix using the cleaned corpus
blogs.clean.tfidf.train = DocumentTermMatrix(blogs.clean.train, control = list(weighting = weightTfIdf))
blogs.clean.tfidf.test = DocumentTermMatrix(blogs.clean.test, control = list(weighting = weightTfIdf))

# remove sparse terms at various thresholds.
# First, we removed the terms that are absent from 99% of the documents.
tfidf.99.train = removeSparseTerms(blogs.clean.tfidf.train, 0.99)  
as.matrix(tfidf.99.train[1:5,1:5])
# Terms
# Docs    “hey “how “i’m “if “it
# 11869    0    0    0   0   0
# 16332    0    0    0   0   0
# 6636     0    0    0   0   0
# 3668     0    0    0   0   0
# 12196    0    0    0   0   0
dim(tfidf.99.train)
# [1] 12880 10876

# There are still too many terms, therefore we remove terms that are absent from 85% of the documents.
tfidf.85.train = removeSparseTerms(blogs.clean.tfidf.train, 0.85)  
as.matrix(tfidf.85.train[1:5, 1:5])
# Terms
# Docs          abil          abl   about    absolut       accept
# 11869 0.0000000000 0.0008607249     0 0.0008164717 0.0000000000
# 16332 0.0003466093 0.0004219853     0 0.0001334298 0.0003833159
# 6636  0.0000000000 0.0006040646     0 0.0011460146 0.0010974200
# 3668  0.0000000000 0.0006951296     0 0.0000000000 0.0012628602
# 12196 0.0000000000 0.0007200442     0 0.0008196288 0.0010464987
dim(tfidf.85.train)
# [1] 12880  1409

tfidf.85.test = removeSparseTerms(blogs.clean.tfidf.test, 0.85)  
as.matrix(tfidf.85.test[1:5, 1:5])
# Terms
# Docs    abil          abl        about absolut      accept
# 4876     0 0.0000000000 0.0000000000       0 0.000000000
# 12227    0 0.0003355524 0.0000000000       0 0.000000000
# 2898     0 0.0000000000 0.0000000000       0 0.000000000
# 12334    0 0.0010111360 0.0007409608       0 0.000901312
# 6489     0 0.0000000000 0.0000000000       0 0.000000000
dim(tfidf.85.test)
# [1] 6440 1393

# Export the terms to csv file
df.tfidf.85.train = as.matrix(tfidf.85.train)
df.tfidf.85.test = as.matrix(tfidf.85.test)
write.csv(df.tfidf.85.train, file = "tfidf85_train.csv")  # The csv file is too large to be uploaded to gitHub
write.csv(df.tfidf.85.test, file = "tfidf85_test.csv")  # The csv file is too large to be uploaded to gitHub
