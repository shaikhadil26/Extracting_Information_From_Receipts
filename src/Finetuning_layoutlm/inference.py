from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2


# load model and processor from huggingface hub
model = LayoutLMForTokenClassification.from_pretrained("shaikhadil26/layoutlm-sroie")
processor = LayoutLMv2Processor.from_pretrained("shaikhadil26/layoutlm-sroie")


# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


label2color = {
    "S-COMPANY": "blue",
    "S-ADDRESS": "yellow",
    "S-DATE": "green",
    "S-TOTAL": "red"
    }


# draw results onto the image
def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image


# run inference
def run_inference(path, model=model, processor=processor, output_image=True):
    # create model input
    image = Image.open(path).convert("RGB")
    encoding = processor(image,
                          max_length=512,
                          padding="max_length",
                          truncation=True,
                          return_tensors="pt")
    del encoding["image"]
    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    
    # Extract words and their corresponding boxes
    words = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    boxes = encoding["bbox"][0]
    
    # Create a list to store extracted information
    extracted_info = {}
    
    # Iterate through words, boxes, and labels together
    for word, box, label in zip(words, boxes, labels):
        # Skip special tokens and 'O' (Outside) labels
        if word in processor.tokenizer.special_tokens_map.values() or label == "O":
            continue
        
        # Remove ## from wordpiece tokens
        word = word.replace("##", "")
        
        # Add to extracted info dictionary
        if label not in extracted_info:
            extracted_info[label] = []
        extracted_info[label].append(word)
    
    print(extracted_info)
    # Join words for each label
    for label in extracted_info:
        extracted_info[label] = " ".join(extracted_info[label])
    print(extracted_info)
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels), extracted_info
    else:
        return extracted_info


image = "/Users/adil/Desktop/Codes/atml_cv/large-receipt-image-dataset-SRD/1016-receipt.jpg"
result, extracted_text = run_inference(image)
result.save("result.png")
result.show()

# Print extracted text with labels
for label, text in extracted_text.items():
    print(f"{label}: {text}")
