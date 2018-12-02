from lxml import etree


def clean(text):
    parts = text.split()
    return ' '.join(parts)


def get_question_text(url):
    from bs4 import BeautifulSoup
    import requests

    page = requests.get(url)  # ship synset

    # BeautifulSoup is an HTML parsing library
    # puts the content of the website into the soup variable, each url on a different line
    soup = BeautifulSoup(page.content, 'html.parser')

    title = soup.find("meta", property="og:title")
    desc = soup.find("meta", property="og:description")

    title = clean(title['content'])
    desc = clean(desc['content'])

    return title, desc


def generate_xml_element(root, title, description):
    thread_elem = etree.Element('Thread')
    rel_q_elem = etree.Element("RelQuestion")

    rel_subject_elem = etree.Element('RelQSubject')
    rel_subject_elem.text = title
    rel_q_elem.append(rel_subject_elem)

    rel_body_elem = etree.Element('RelQBody')
    rel_body_elem.text = description
    rel_q_elem.append(rel_body_elem)

    thread_elem.append(rel_q_elem)
    root.append(thread_elem)


def prettify(elem):
    from xml.etree import ElementTree
    from xml.dom import minidom
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def main():
    question_urls = []
    with open('data/ydata-yanswers-question-types-sample_of_1000-v1_1.txt', 'r') as file:
        lines = file.readlines()

        for line in lines:
            parts = line.split('\t')
            cat, url = parts[0], parts[1]
            question_urls.append((cat, url))

    with open('data/yahoo-questions.xml', 'w') as xml_file, open('data/yahoo-questions.txt', 'w') as txt_file:
        root = etree.Element('root')
        for cat, url in question_urls:
            title, desc = get_question_text(url)
            if len(title) is not 0:
                generate_xml_element(root, title, desc)
                txt_file.write(cat + "\t" + title + " " + desc + "\n")
        xml_file.write(prettify(root))


if __name__ == "__main__":
    main()
