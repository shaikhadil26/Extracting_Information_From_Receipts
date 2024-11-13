import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io
import tempfile
import os
from helper.preprocess_img import preprocess_image
from helper.inference_img import extract_receipt_info

# Set up the Streamlit app
st.title("Receipt Information Extractor")
st.write("Upload your receipt images to extract key information.")

def save_uploaded_file(uploadedfile):
    """Save uploaded file temporarily and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploadedfile.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_receipt(uploaded_file):
    """Process the receipt image and extract information."""
    try:
        # Save the uploaded file temporarily
        temp_path = save_uploaded_file(uploaded_file)
        if temp_path is None:
            return None
            
        try:
            # Preprocess the image using the file path
            preprocessed_image = preprocess_image(temp_path)  # This returns a PIL Image
            
            # Save the preprocessed image to a new temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                preprocessed_path = tmp_file.name
                preprocessed_image.save(preprocessed_path, 'JPEG')
            
            # Extract information using the LayoutLM model with the preprocessed image path
            receipt_info = extract_receipt_info(preprocessed_path)
            
            # Clean up the temporary files
            os.unlink(temp_path)
            os.unlink(preprocessed_path)
            
            return receipt_info
            
        except Exception as e:
            st.error(f"Error in model inference: {str(e)}")
            # Clean up the temporary file
            os.unlink(temp_path)
            if 'preprocessed_path' in locals():
                try:
                    os.unlink(preprocessed_path)
                except:
                    pass
            return None
            
    except Exception as e:
        st.error(f"Error processing receipt: {str(e)}")
        return None

def main():
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose receipt images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create a list to store receipt data
        receipts_data = []
        total_amount = 0.0
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Create columns for image and info
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display the original image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                
                with col2:
                    # Process the receipt
                    with st.spinner(f'Processing {uploaded_file.name}...'):
                        receipt_info = process_receipt(uploaded_file)
                    
                    if receipt_info:
                        # Extract information (using get() to handle missing keys)
                        receipt_data = {
                            'Image Name': uploaded_file.name,
                            'Company': receipt_info.get('company', ''),
                            'Address': receipt_info.get('address', ''),
                            'Date': receipt_info.get('date', ''),
                            'Total': receipt_info.get('total', '0.00')
                        }
                        
                        # Add to total amount if total is present
                        try:
                            if receipt_data['Total']:
                                total_amount += float(receipt_data['Total'])
                        except ValueError:
                            st.warning(f"Could not parse total amount for {uploaded_file.name}")
                        
                        receipts_data.append(receipt_data)
                        st.success(f"Successfully processed {uploaded_file.name}")
                    else:
                        st.warning(f"Could not extract information from {uploaded_file.name}")
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(uploaded_files))
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Display results in a table if we have data
        if receipts_data:
            # Create DataFrame
            df = pd.DataFrame(receipts_data)
            
            # Display the table
            st.subheader("Extracted Receipt Information")
            st.dataframe(df, use_container_width=True)
            
            # Display total amount
            st.markdown("---")
            st.markdown(f"**Total Amount:** ${total_amount:.2f}")
        else:
            st.warning("No receipt information could be extracted from the uploaded images.")

if __name__ == "__main__":
    main()