# %%
import xml.etree.ElementTree as ET
import json
import datasets
from pathlib import Path
from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor
from transformers.file_utils import PaddingStrategy
from dataclasses import dataclass
import torch
from typing import Optional, Union
import numpy as np
from PIL import Image

def normalize_bbox(bbox, width, height, scale=1000):
     return [
        max(int(scale * (bbox[0] / width)), 0),
        max(int(scale * (bbox[1] / height)), 0),
        min(int(scale * (bbox[2] / width)), scale),
        min(int(scale * (bbox[3] / height)), scale),
     ]

def unnormalize_bbox(bbox, width, height, scale=1000):
    return [
        int((bbox[0] / scale) * width),
        int((bbox[1] / scale) * height),
        int((bbox[2] / scale) * width),
        int((bbox[3] / scale) * height),
    ]

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
    for page in root.findall(".//"+xlmns+"Page"):
        # Extract page dimensions
        page_width = int(page.attrib["WIDTH"])
        page_height = int(page.attrib["HEIGHT"])

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

# %%
def overlap(box1, box2):
  """
  This function computes the area of overlap between two bounding boxes.

  Args:
      box1: A list containing the coordinates of the first bounding box (xmin, ymin, xmax, ymax)
      box2: A list containing the coordinates of the second bounding box (xmin, ymin, xmax, ymax)

  Returns:
      The percentage [0.0-1.0] of the area of the first bounding box that is overlapped by the second bounding box.
  """

  assert box1[0] <= box1[2]
  assert box1[1] <= box1[3]
  assert box2[0] <= box2[2]
  assert box2[1] <= box2[3]

  if box1[0] == box1[2] or box1[1] == box1[3] or box2[0] == box2[2] or box2[1] == box2[3]:
      return 0.0

  # determine the coordinates of the intersection rectangle
  x_left = max(box1[0], box2[0])
  y_top = max(box1[1], box2[1])
  x_right = min(box1[2], box2[2])
  y_bottom = min(box1[3], box2[3])

  if x_right < x_left or y_bottom < y_top:
      return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / bb1_area
  assert iou >= 0.0
  assert iou <= 1.0
  return iou


def is_inside(box1, box2):
  """
  This function checks if box1 is inside box2.

  Args:
      box1: A list containing the coordinates of the first bounding box (xmin, ymin, xmax, ymax)
      box2: A list containing the coordinates of the second bounding box (xmin, ymin, xmax, ymax)

  Returns:
      True if box1 is inside box2, False otherwise.
  """
  x_left1, y_top1, x_right1, y_bottom1 = box1
  x_left2, y_top2, x_right2, y_bottom2 = box2
  return x_left1 >= x_left2 and y_top1 >= y_top2 and x_right1 <= x_right2 and y_bottom1 <= y_bottom2

# %%


def label_tokens(tokens, bboxes, annotation):
    """
    This function labels the tokens extracted from an ALTO file with the corresponding label from the datast based on the label bounding box.

    Args:
        tokens: A list of strings, where each string is the content of a <String> element in the ALTO file.
        bboxes: A list of lists of integers, where each inner list represents the bounding box of the corresponding token in the tokens list.
        annotation: A list of annotations for the image.
    Returns:
        3 lists, first containing the tokens, second containing the bounding boxes, and third containing the corresponding label.
    """
    labels = []
    for i, bbox in enumerate(bboxes):
        for label in annotation:
            l =  label["labels"][0]
            orig_w, orig_h = label["original_width"], label["original_height"]
            if label.get('points') is not None:
                x, y, x2, y2 = polygon_to_bbox(label["points"])
                x, y , x2, y2 = int(x * orig_w/100), int(y * orig_h/100), int(x2 * orig_w/100), int(y2 * orig_h/100)
                label_bbox = [x, y, x2, y2]
            elif label.get('x') is not None:
                x, y = label["x"], label["y"]
                w, h = label["width"], label["height"]
                x, y , w, h = int(x * orig_w/100), int(y * orig_h/100), int(w * orig_w/100), int(h * orig_h/100)
                label_bbox = [x, y, x + w, y + h]
            else:
                print(f"Unsuppoted label format. Skipping: {label}")
                continue
            
            if is_inside(bbox, label_bbox) or overlap(bbox, label_bbox) > 0.4:
                labels.append(l)
                break
        if len(labels) != i + 1:
            labels.append("background")

    return tokens, bboxes, labels

def polygon_to_bbox(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return [min(x), min(y), max(x), max(y)]

label_names = ["background", "birth_date", "death_book", "death_date", "funeral_date", "grave_id", "grave_location", "information", "information_source", "key", "name", "nationality", "rank"]
label_to_ner = {"background": "O",
                "birth_date": "BDATE",
                "death_book": "DEATHBOOK",
                "death_date": "DEATHDATE",
                "funeral_date": "FUNERAL",
                "grave_id": "GRAVEID",
                "grave_location": "GRAVELOC",
                "information": "INFO",
                "information_source": "INFOSRC",
                "key": "KEY",
                "name": "NAME",
                "nationality": "NAT",
                "rank": "RANK"}
# create list of ner tags from label_to_ner.values() 
ner_tags = ["O"]
for label in set(label_to_ner.values()) - {"O"}:
    ner_tags.append("B-" + label)
    ner_tags.append("I-" + label)
ner_tags

ner2id = {ner: i for i, ner in enumerate(ner_tags)}
id2ner = {i: ner for i, ner in enumerate(ner_tags)}

def convert_labels_to_ner(labels, label_to_ner):
    """
    This function converts the labels to NER tags.
      From processing of the ALTO OCR we have tokens ordered correctly,
        so we can just convert the continuous label groups to NER tags with B- and I- marks respectively.

    Args:
        labels: A list of labels.
        label_to_ner: A dictionary mapping labels to NER tags.

    Returns:
        A list of NER tags from labels.
    """
    ner_tags = []
    prev_label = None
    for label in labels:
        if label == "background":
            ner_tags.append(label_to_ner[label])
        else:
            if prev_label == label:
                ner_tags.append(f"I-{label_to_ner[label]}")
            else:
                ner_tags.append(f"B-{label_to_ner[label]}")
        prev_label = label
    return ner_tags


class VHA(datasets.DatasetBuilder):
    """
    A dataset builder for the VHA dataset. Combining the ALTO OCR data for each document image with the labels from the CSV file.
    Args:
        config: A `datasets.BuilderConfig` object.
        cache_dir: A string representing the cache directory to store the downloaded and processed data.
    """

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features({
                "file_id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                "labels": datasets.Sequence(datasets.Value("string")),
                "image": datasets.Value("string"),
                "alto": datasets.Value("string")
            })
        )

    def _split_generators(self, dl_manager):
        """Defines splits in the dataset (e.g., train, dev, test)."""
        urls = {
            "train": "", 
            "dev": ""
        }
        data_files = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(name=split_name, gen_kwargs={"data_file": data_files[split_name]})
            for split_name in self.config.split
        ]

    def _generate_examples(self, data_file):
        """Parses data from downloaded files into examples."""
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                # Parse each line based on your data format
                text, label = line.strip().split("\t")  # Example for tab-separated data
                yield {
                    "text": text,
                    "label": label
                }

# Define dataset configurations (optional)
class ConfigVHA(datasets.BuilderConfig):
    """Builder configuration options (e.g., version, features)."""

    name = "vha_config"
    version = "1.0.0"
    ignore_supervised_keys = False  # Set to True if features are not for supervised learning


def tokenize_and_expand_features(tokens, bounding_boxes, labels, tokenizer):
    """
    This function tokenizes the tokens and expands the labels to match the tokenization.

    Args:
        tokens: A list of tokens.
        bounding_boxes: A list of bounding boxes for each token.
        labels: A list of labels for each token.
        tokenizer: A tokenizer object to tokenize the tokens.

    Returns:
        A list of tokenized tokens, bounding boxes, and labels.
    """
    assert len(tokens) == len(bounding_boxes) == len(labels), f"The number of tokens, bounding boxes, and labels must match. {len(tokens)=} != {len(bounding_boxes)=} != {len(labels)=}"
    return tokenizer(tokens, boxes=bounding_boxes, word_labels=labels, padding=False, truncation=False)


def preprocess_data(examples, tokenizer=None, dataset_dir=None, image_dir=None, alto_dir=None, image_size=(224, 224)):
    """
    Original data for this dataset consists of just the images of documents.
    Those images were labeld using Label Studio and the output is stored in a CSV file.
    Annotated data is stored in the CSV file with the following important columns:
        - ocr: The name of the image file (weird)
        - transcription: The transcription of the image (if annotated with transcriptions, most of the time empty)
        - label: List of dictionaries where each contains annotation bounding box with it's label and original image size.
    Another important file is the ALTO file for each image, which contains the OCR data.
    The ALTO file is processed to extract tokens and bounding boxes for each token which is then labeled with the label from the CSV file.
    """
    assert dataset_dir is not None, "Please provide the dataset directory."
    assert image_dir is not None, "Please provide the image directory."
    assert alto_dir is not None, "Please provide the ALTO directory."
    assert tokenizer is not None, "Please provide a tokenizer."

    dataset_dir = Path(dataset_dir)
    alto_dir = Path(alto_dir)
    image_dir = Path(image_dir)
    # find the original image name 
    # print(examples.head())
    iter_examples = list(examples.to_dict("records"))
    file_names = [example["ocr"].rsplit("-")[-1] for example in iter_examples]
    for file in file_names:
        alto_file = alto_dir / file.replace(".jpg", ".xml")
        image_file = image_dir / file
        image = Image.open(image_file)
        orig_width, orig_height = image.size
        image = image.resize(image_size)
        image = image.convert("RGB")
        tokens, bounding_boxes = load_and_process_alto(alto_file) # extracting tokens and bounding boxes from ALTO file
        annot = [json.loads(value) for value in examples[examples["file"] == file]["label"].values][0] # extracting annotations from CSV file
        tokens, bounding_boxes, labels = label_tokens(tokens, bounding_boxes, annot) # labeling tokens with annotations
        bounding_boxes = [normalize_bbox(bbox, orig_width, orig_height) for bbox in bounding_boxes] # normalizing bounding boxes
        ner_tags = convert_labels_to_ner(labels, label_to_ner) # converting labels to NER tags
        yield {"tokens": tokens, "boxes": bounding_boxes, "labels": ner_tags, "image": image}



@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["image"] for feature in features], return_tensors="pt").pixel_values
        # prepare text input
        for feature in features:
            del feature["image"]
        tokens = [feature["tokens"] for feature in features]
        boxes = [feature["boxes"] for feature in features]
        labels = [[ner2id[label] for label in feature["labels"]] for feature in features]
        batch = self.tokenizer(
            text= tokens,
            boxes= boxes,
            word_labels= labels,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch["image"] = image
            
        return batch




    


