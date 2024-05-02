A B S T R A C T 

This database is provided for the Fake News Detection task. In addition to being used in other tasks of detecting fake news, it can be specifically used to detect fake news using the Natural Language Inference (NLI).

Samples of this data set are prepared in two steps. In the first step, the existing samples of the PoliticFact.Com website have been crawled using the API until April 26. This website is a reputable source of fact-finding in which a news team and fact-finding experts evaluate political news articles published in various sources (CNN, BBC, Facebook). In the second step, a preprocessing has been performed on the crawled data to remove those parts of the text which represent include the actual veracity label of the news.

Instructions:

This dataset is designed and stored to be compatible for use with both the LIAR test dataset and FakeNewsNet (PolitiFact) datasets as evaluation data. There are two folders, each containing three CSV files.

1. 15212 training samples, 1058 validation samples, and 1054 test samples are the same as (FakeNewsNet PolitiFact) data. The classes of this data are ”real” and ”fake”.

2. 15052 training samples, 1265 validation samples, and 1266 test samples, which is the same as the LIAR test data. The classes in this data are ”pants-fire”, ”false”, and ”barely true”, ”half-true”, ”mostly-true” and ”true”.

The DataSet columns:

id      : matches the id in the PolitiFact website API (unique for each sample)

date    : The time each article was published in the PolitiFact website

speaker  : The person or organization to whom the Statement relates

statement  : A claim published in the media by a person or an organization and has been investigated in the PolitiFact article.

sources   : The sources used to analyze each Statement

paragraph_based_content  : content stored as paragraphed in a list

fullText_based_content  : Full text using pasted paragraphs

label  : The class for each sample
