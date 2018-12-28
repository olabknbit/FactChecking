import serpscrap


def clean_data(data):
    import re

    REPLACE_NO_SPACE = re.compile("(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(\])|(=)|(_)|(\*)|(://)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(-)|(/)(&\s\s\s)")

    def preprocess_reviews(text):
        text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
        text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]

        return ''.join(text)

    return preprocess_reviews(data)


def clean_bag_of_words_stop_words(data):
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words("english"))
    data = clean_data(data)
    bag_of_words = []
    for text in data.split():
        words_filtered = [e.lower() for e in text.split() if len(e) >= 3]

        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        bag_of_words.append(words_without_stopwords)

    return bag_of_words


def get_scrapes(keyword):
    query = clean_bag_of_words_stop_words(keyword)
    query = ' '.join([item for sublist in query for item in sublist])
    keywords = query
    config = serpscrap.Config()
    config.set('scrape_urls', False)

    scrap = serpscrap.SerpScrap()
    scrap.init(config=config.get(), keywords=keywords)
    results = scrap.run()

    rrs = []
    for result in results:
        rrs.append(result)

    def strip(obj):
        return obj if obj is not None else ' '

    return ' '.join([strip(result['serp_snippet']) + strip(result['serp_title']) for result in results])


def write_snippets_to_file(mode, start_from=None):
    works = True
    if start_from is not None:
        works = False
    with open('data/b/b-factual-q-a-clean-' + mode + '.txt', 'r') as f:
        lines = f.readlines()
        last_question = ''
        queries = []
        for line in lines:
            prts = line.split('\t')
            tag = prts[0]
            typ = prts[1]
            text = prts[2]
            if typ == 'NA':
                last_question = text
            else:
                query = last_question + ' ' + text
                queries.append((tag, query))

    with open('data/b/b-factual-q-a-web-results-' + mode + '.txt', 'a') as f:
        for tag, query in queries:
            if works:
                snippets = get_scrapes(query)
                f.write(tag + '\t' + snippets + '\n')
            if tag == start_from:
                works = True


if __name__ == '__main__':
    import nltk

    nltk.download('stopwords')
    write_snippets_to_file('test')
    write_snippets_to_file('train')
