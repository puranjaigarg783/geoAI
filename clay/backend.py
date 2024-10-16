# backend.py

import os
import json
import groq
import re
import datetime
from dateutil import parser
import importlib.util
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key from .env file
client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))

def extract_info_from_query(query):
    # Call LLaMA model through Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that extracts information from queries about natural disasters. "
                    "Extract the location, start date, and end date from the query. If no specific dates are mentioned, use the current date as the end date and 14 days before as the start date. "
                    "Always provide the location name, even if it's just a state or country. Return the information in a structured format like this:\n"
                    "Location: [extracted location]\n"
                    "Start Date: [extracted start date or 'Not specified']\n"
                    "End Date: [extracted end date or 'Not specified']"
                )
            },
            {
                "role": "user",
                "content": f"Extract location_name, start_date, and end_date from this query: {query}"
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=200
    )

    # Extract the response
    response = chat_completion.choices[0].message.content

    # Parse the response to extract information
    info = {}
    for line in response.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip().lower()] = value.strip()

    # Clean up location name
    if 'location' in info:
        # Remove any date-related words from the location
        date_words = ['past', 'last', 'previous', 'recent', 'week', 'month', 'year', 'day']
        location_parts = info['location'].split()
        info['location'] = ' '.join(word for word in location_parts if word.lower() not in date_words)
        
        # Remove any trailing "in the" or similar phrases
        info['location'] = re.sub(r'\s+in\s+the\s*$', '', info['location'], flags=re.IGNORECASE)

    # Ensure location_name is not empty
    if 'location' not in info or not info['location']:
        # If location is not extracted, attempt to find it in the original query
        location_match = re.search(r'in\s+(\w+(?:\s+\w+)*)', query, re.IGNORECASE)
        if location_match:
            info['location'] = location_match.group(1)
        else:
            info['location'] = "Unknown location"

    # Convert dates to proper format and handle relative dates
    today = datetime.date.today()
    end_date_str = info.get('end date', '').lower()
    start_date_str = info.get('start date', '').lower()

    if end_date_str == 'not specified' or not end_date_str:
        info['end_date'] = today.isoformat()
    else:
        try:
            info['end_date'] = parser.parse(end_date_str).date().isoformat()
        except:
            info['end_date'] = today.isoformat()

    if start_date_str == 'not specified' or not start_date_str:
        info['start_date'] = (today - datetime.timedelta(days=14)).isoformat()
    else:
        try:
            info['start_date'] = parser.parse(start_date_str).date().isoformat()
        except:
            info['start_date'] = (today - datetime.timedelta(days=14)).isoformat()

    return {
        "location_name": info['location'],
        "start_date": info['start_date'],
        "end_date": info['end_date']
    }

def save_to_json(data, filename='extracted_params.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def run_prediction_script():
    # Import the prediction_script module
    spec = importlib.util.spec_from_file_location("prediction_script", "prediction_script.py")
    prediction_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prediction_module)

    # Run the main method of prediction_script
    prediction_module.main()
