# FactChecking
## Uni project solving SemEval 2019 Task 8 on Fact-Checking in Community Forums.

Problem description can be found: https://competitions.codalab.org/competitions/20022

Our main inspiration/reference is https://arxiv.org/pdf/1803.03178.pdf and their github repo: https://github.com/qcri/QLFactChecking

## Problem short description
There are two subtask in this problem:

### Subtask A
Question classification: determine which class (factual, opinion, socializing) does the question belong to.

Our approach: SVM (probably)

### Subtask B
Answer factuality determination: classify each answer into one of three classes: factual-true, factual-false, non-factual.

Out approach: Based on Fact-checking in community forums by Mihaylova, Nakov (2018), we will use multifaceted model. Our model will address language used in answer; context in which answer is located; external sources support.
Answer content features include subjectivity and credibility analysis
Answer’s context is computed based on cosine similarity to other answers in the current thread
Web support is based on automatic queries which are used to compute cosine similarity with question-answer pair


### Used datasets
For subtask A, we're using Yahoo Answers datasets from Webscope (cannot be posted on github) 

1. L31 - Questions on Yahoo Answers labeled as either informational or conversational, version 1.0
2. L9 - Yahoo! Answers Question Types Sample of 1000, version 1.0

For subtask B, we're using a dataset created by authors of https://arxiv.org/pdf/1803.03178.pdf, called QL-factual-questions.xml