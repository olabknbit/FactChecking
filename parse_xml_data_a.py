import xml.etree.ElementTree as ET
import pandas as pd
import string

if __name__ == "__main__":
    file = "questions_train.xml"
    tree = ET.parse(file)
    root = tree.getroot()

    ids = []
    labels = []
    questions = []

    for item in root.findall('./Thread/RelQuestion'):

        ids.append(str(item.attrib['RELQ_ID']))
        labels.append(str(item.attrib['RELQ_FACT_LABEL']).lower())

        text = ''
        for q in item:
            if q.text is not None:
                text = text + ' ' + str(q.text)
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        questions.append(text)

    results = pd.DataFrame(list(zip(ids, questions)))
    results.to_csv('sem-train.txt', sep='\t', header=None, index=None)
