import streamlit as st
import os
import json
from backend import extract_info_from_query, save_to_json, run_prediction_script

def main():
    st.title("GeoAI Clay Demo")

    # Get user query
    query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if query.strip() == "":
            st.warning("Please enter a query.")
            return

        # Extract information from the query
        extracted_info = extract_info_from_query(query)

        if not extracted_info.get('location_name'):
            st.error("Error: Could not extract location name from the query.")
            return

        # Display extracted information
        st.write("**Extracted Information:**")
        st.write(f"**Location:** {extracted_info['location_name']}")
        st.write(f"**Start Date:** {extracted_info['start_date']}")
        st.write(f"**End Date:** {extracted_info['end_date']}")

        # Save extracted information to JSON file
        save_to_json(extracted_info)

        # Run the prediction script
        with st.spinner("Processing..."):
            run_prediction_script()

        # Display the output images and predictions
        output_image_dir = "output_images"
        output_metadata_file = "output_metadata.json"

        if os.path.exists(output_metadata_file):
            with open(output_metadata_file, 'r') as f:
                metadata_list = json.load(f)

            if len(metadata_list) == 0:
                st.warning("No images were generated.")
            else:
                for metadata in metadata_list:
                    image_path = os.path.join(output_image_dir, metadata['image_filename'])
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"{metadata['prediction_label']} on {metadata['date']} at {metadata['location']}", use_column_width=True)
                    else:
                        st.warning(f"Image {metadata['image_filename']} not found.")
        else:
            st.error("No metadata file found. Please check the backend processing.")

if __name__ == "__main__":
    main()
