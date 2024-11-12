from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2

model = LayoutLMForTokenClassification.from_pretrained("shaikhadil26/layoutlm-sroie")
processor = LayoutLMv2Processor.from_pretrained("shaikhadil26/layoutlm-sroie")


def extract_receipt_info(image_path):
    """
    Extract information from a receipt image using LayoutLM model.
    
    Args:
        image_path (str): Path to the receipt image
        
    Returns:
        dict: Dictionary containing extracted information
    """
    try:
        # Process image
        image = Image.open(image_path).convert("RGB")
        encoding = processor(image,
                           max_length=512,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        
        # Remove image from encoding as it's not needed for inference
        del encoding["image"]
        
        # Run inference
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = [model.config.id2label[prediction] for prediction in predictions]
        
        # Get words and boxes
        words = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        boxes = encoding["bbox"][0]
        
        # Extract information
        extracted_info = {}
        current_total_tokens = []
        previous_label = None
        
        # Process tokens
        for word, box, label in zip(words, boxes, labels):
            if word in processor.tokenizer.special_tokens_map.values():
                continue
                
            word = word.replace("##", "")
            
            # Special handling for total amounts
            if label == "S-TOTAL" or (label == "O" and previous_label == "S-TOTAL" and any(c.isdigit() or c == '.' for c in word)):
                current_total_tokens.append(word)
            
            if label not in extracted_info and label != "O":
                extracted_info[label] = []
            if label != "O":
                extracted_info[label].append(word)
                
            previous_label = label
        
        # Process extracted information
        result = {
            'company': '',
            'address': '',
            'date': '',
            'total': ''
        }
        
        for label in extracted_info:
            if label == "S-TOTAL":
                # Include any additional total tokens
                all_total_tokens = extracted_info[label] + [t for t in current_total_tokens if t not in extracted_info[label]]
                total_str = "".join(all_total_tokens)
                # Clean up total amount
                total_str = ''.join(c for c in total_str if c.isdigit() or c == '.')
                if '.' not in total_str and len(total_str) > 2:
                    total_str = total_str[:-2] + '.' + total_str[-2:]
                result['total'] = total_str
            elif label == "S-COMPANY":
                result['company'] = " ".join(extracted_info[label])
            elif label == "S-ADDRESS":
                result['address'] = " ".join(extracted_info[label])
            elif label == "S-DATE":
                result['date'] = " ".join(extracted_info[label])
        
        return result
        
    except Exception as e:
        print(f"Error processing receipt: {str(e)}")
        return None