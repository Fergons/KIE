import xml.etree.ElementTree as ET

def transform_to_pixels(bbox, page_margins):
    """
    Transforms the bounding box coordinates from ALTO's relative units to pixels.
    """
    x1, y1, x2, y2 = bbox
    page_width, page_height = page_margins
    x1 = int(x1 * page_width)
    x2 = int(x2 * page_width)
    y1 = int(y1 * page_height)
    y2 = int(y2 * page_height)
    return [x1, y1, x2, y2]

def load_and_process_alto(alto_file):
    """
    Loads and processes an ALTO file, returning two lists:
        tokens: a list of strings, where each string is the content of a <String> element
        bboxes: a list of lists of integers, where each inner list represents the bounding box
                of the corresponding token in the tokens list. The format is [X1, Y1, X2, Y2]
    """
    tokens = []
    bboxes = []
    page_margins = []
    tree = ET.parse(alto_file)
    root = tree.getroot()
    # get xlmns from alto file
    xlmns = root.tag.split('}')[0] + '}'
    # Find all <Page> elements
    for page in root.findall(".//"+xlmns+"PrintSpace"):
        # Extract page dimensions
        page_width = int(page.attrib["WIDTH"])
        page_height = int(page.attrib["HEIGHT"])
        page_margins = [page_width, page_height]
        # Find all <TextLine> elements
        for textline in page.findall(".//"+xlmns+"TextLine"):
            for string in textline.findall(".//"+xlmns+"String"):
                # Extract token text and bounding box coordinates
                content = string.attrib["CONTENT"]
                h = int(string.attrib["HEIGHT"])
                w = int(string.attrib["WIDTH"])
                vpos = int(string.attrib["VPOS"])
                hpos = int(string.attrib["HPOS"])
                x1, y1 = hpos, vpos
                x2, y2 = x1 + w, y1 + h

                tokens.append(content)
                bboxes.append([x1, y1, x2, y2])

    return tokens, bboxes


