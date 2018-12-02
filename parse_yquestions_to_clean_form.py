def word_clean(word):
    return word.lower()


def clean(text):
    import re
    apos = '&#39;'
    quot = '&quot;'
    # we want to keep the contractions
    text = text.replace(apos, '\'')
    # but delete quotes
    text = text.replace(quot, '')

    p = re.compile('(&.+;)')
    text = p.sub(' ', text)

    words = text.split()
    words = [word_clean(word) for word in words]
    text = ' '.join(words)
    return text


def main():
    cat_mappings = {'advice': 'opinion', 'informational': 'factual', 'conversational': 'socializing'}
    with open('data/yahoo-questions.txt', 'r') as input_file, open('data/yahoo-question-clean.txt', 'w') as output_file:
        lines = input_file.readlines()

        q_id = 0
        for line in lines:
            parts = line.split('\t')
            cat, question = parts[0], parts[1]
            if cat in cat_mappings.keys():
                question = clean(question)
                output_file.write(str(q_id) + '\t' + cat_mappings[cat] + '\t' + question + '\n')
                q_id += 1


if __name__ == "__main__":
    main()