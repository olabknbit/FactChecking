PREFIX = 'data/'
OUT_FILENAME1000 = 'yahoo-question-clean.txt'
IN_FILENAME1000 = 'yahoo-questions.txt'
IN_FILENAME4000_1 = 'batch1.csv'
IN_FILENAME4000_2 = 'batch2.csv'
OUT_FILENAME4000 = 'batch1+2-clean.txt'
OUT_FILENAME5000 = 'all-yquestions-clean.txt'


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


def parse_data1000(cat_mappings):
    with open(PREFIX + IN_FILENAME1000, 'r') as input_file, open(PREFIX + OUT_FILENAME1000, 'w') as output_file:
        lines = input_file.readlines()

        q_id = 0
        for line in lines:
            parts = line.split('\t')
            cat, question = parts[0], parts[1]
            if cat in cat_mappings.keys():
                question = clean(question)
                output_file.write(str(q_id) + '\t' + cat_mappings[cat] + '\t' + question + '\n')
                q_id += 1


def parse_data4000(cat_mappings):
    with open(PREFIX + IN_FILENAME4000_1, 'r', encoding='latin-1') as input_file1, \
            open(PREFIX + IN_FILENAME4000_2, 'r', encoding='latin-1') as input_file2, \
            open(PREFIX + OUT_FILENAME4000, 'w') as output_file:
        lines1 = input_file1.readlines()
        lines2 = input_file2.readlines()
        q_id = 1000
        for line in lines1 + lines2:
            line.encode('utf-8').strip()
            parts = line.split(',')
            title, description, category, top, url, label = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            label = label.strip()
            if label in cat_mappings.keys():
                if description.lower() == 'null':
                    description = ''
                question = clean(title + description)
                output_file.write(str(q_id) + '\t' + cat_mappings[label] + '\t' + question + '\n')
                q_id += 1


def combine():
    with open(PREFIX + OUT_FILENAME1000, 'r') as input_file1, \
            open(PREFIX + OUT_FILENAME4000, 'r') as input_file2, \
            open(PREFIX + OUT_FILENAME5000, 'w') as output_file:
        lines1 = input_file1.readlines()
        lines2 = input_file2.readlines()
        output_file.writelines(lines1 + lines2)


def main():
    cat_mappings = {'advice': 'opinion', 'informational': 'factual', '1': 'socializing', '0': 'factual'}
    parse_data1000(cat_mappings)
    parse_data4000(cat_mappings)
    combine()


if __name__ == "__main__":
    main()