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

    with open('data/b/b-factual-q-a-web-results-' + mode + '.txt', 'w') as f:
        for tag, query in queries:
            if works:
                snippets = get_scrapes(query)
                f.write(tag + '\t' + snippets + '\n')
            if tag == start_from:
                works = True


def fix_snippets(mode):
    with open('data/b/b-factual-q-a-clean-' + mode + '.txt', 'r') as f:
        lines = f.readlines()
        last_question = ''
        queries = []
        for line in lines:
            prts = line.split('\t')
            tag = prts[0].strip()
            typ = prts[1]
            text = prts[2].strip()
            if typ == 'NA':
                last_question = text
            else:
                query = last_question + ' ' + text
                queries.append((tag, query))

    save_filename = 'data/b/b-factual-q-a-web-results-' + mode + '.txt'
    with open(save_filename + '.copy', 'r') as f:
        lines = f.readlines()
        dones = []
        for line in lines:
            prts = line.split('\t')

            if len(prts) == 1:
                tag = prts[0].strip()
                dones.append((tag, None))
            if len(prts) > 1:
                tag = prts[0].strip()
                text = prts[1].strip()
                dones.append((tag, text))

    with open(save_filename, 'w') as f:
        for q_row, done_row in zip(queries, dones):
            q_tag, query = q_row
            d_tag, snippets = done_row
            if q_tag != d_tag:
                print("OOHH", q_tag, d_tag, len(q_tag), len(d_tag))
                exit(1)
            # if snippets == '':
            #     snippets = get_scrapes(query) + '\n'
            # print(snippets)
            if snippets is None:
                snippets = get_scrapes(query)
            f.write(q_tag + '\t' + snippets + '\n')


def scrape_questions(mode):
    with open('data/b/b-factual-q-a-clean-' + mode + '.txt', 'r') as f:
        lines = f.readlines()
        queries = []
        for line in lines:
            prts = line.split('\t')
            tag = prts[0]
            typ = prts[1]
            text = prts[2]
            if typ == 'NA':
                query = text
                queries.append((tag, query))

    with open('data/b/b-factual-q-web-results-' + mode + '.txt', 'w') as f:
        for tag, query in queries:
            snippets = get_scrapes(query)
            f.write(tag + '\t' + snippets + '\n')


if __name__ == '__main__':
    import nltk

    nltk.download('stopwords')
    # write_snippets_to_file('test')
    # write_snippets_to_file('train')
    # fix_snippets('train')
    scrape_questions('train')
    scrape_questions('test')
