from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** "
    "rating, it captures the salt-air energy of the coast through its signature "
    "beer-battered snapper tacos and zesty octopus ceviche, making it a premier "
    "spot for open-air dining near the pier. Price range: $"
)

EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
    }}
"""

MAX_REPAIR_ATTEMPTS = 3


## Exercise 1: Integrate the LLM model from Lesson 1

#Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = """
    You are a precise data extraction assistant. Your job is to read an unstructured
    restaurant description and convert it into a single, strictly valid JSON object.

    Rules:
    - Output ONLY the JSON object. No explanations, no markdown code fences, no extra text.
    - Use exactly these keys: name, location, type, food_style, rating, price_range,
      signatures, vibe, environment, shortcomings.
    - "name": the restaurant's name as a string.
    - "location": the restaurant's location/neighborhood as a string.
    - "type": the restaurant type/category as a string (e.g., "casual taqueria").
    - "food_style": the cuisine or food style as a string.
    - "rating": the numeric rating as a float (e.g., 4.2). If no rating is given, use null.
    - "price_range": convert any dollar-sign notation (e.g., $, $$, $$$) into an integer
      equal to the number of dollar signs. If no price information is given, use null.
    - "signatures": a list of strings naming the signature/hero dishes. Use an empty
      list if none are mentioned.
    - "vibe": a short string describing the restaurant's atmosphere/vibe.
    - "environment": a string describing the restaurant's setting/environment.
    - "shortcomings": a list of strings describing any noted shortcomings or
      drawbacks. Use an empty list if none are mentioned.
    - Do not invent information that is not present or implied in the text.
    """

    base_user_prompt = f"""
    Task:
    Extract the structured attributes from the restaurant description below and
    return them as a single JSON object following exactly the schema and
    formatting style shown in the example.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

    """
    return base_system_msg, base_user_prompt


# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt, params=None):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    response = model.chat(messages=messages)
    output_text = response["choices"][0]["message"]["content"]

    return output_text


def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = """
    You are a JSON repair assistant. You will be given a candidate JSON output that
    failed schema validation, along with the validation error message. Your sole
    task is to fix the candidate JSON so that it becomes strictly valid and conforms
    to the schema implied by the error message.

    Rules:
    - Output ONLY the corrected JSON object. No explanations, no markdown code
      fences, no extra commentary.
    - Preserve all correct information from the candidate JSON; only fix what is
      broken (e.g., missing/extra keys, wrong types, malformed syntax such as
      missing commas or unbalanced brackets).
    - Do not invent new factual content that was not present in the candidate JSON.
    """
    auto_repair_prompt = f"""
    Task:
    Fix the following candidate JSON output so that it resolves the validation
    error below and becomes strictly valid JSON.

    Candidate JSON output:
    {response}

    Validation error message:
    {error_message}

    Return only the corrected JSON object.
    """
    return auto_repair_system_msg, auto_repair_prompt


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def new_data_entry_process(paragraph, itemId):
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate_json_output = llm_model(base_system_msg, base_user_prompt)

    attempt = 0
    while True:
        try:
            Restaurant.model_validate_json(candidate_json_output)
            break
        except ValidationError as e:
            attempt += 1
            if attempt > MAX_REPAIR_ATTEMPTS:
                raise
            error_message = e.json()
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(candidate_json_output, error_message)
            candidate_json_output = llm_model(repair_system_msg, repair_prompt)

    structured_entry = json.loads(candidate_json_output)
    structured_entry['itemId'] = itemId
    return structured_entry


def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    print(f"\n--- Record #{index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")


def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                print(f"{i}: {res.get('name', 'N/A')}")

        elif choice == '2':
            index = input("Enter record index: ")
            if index.isdigit() and 0 <= int(index) < len(data):
                show_restaurant_card(data[int(index)], int(index))
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1 #the item id for the new data

                paragraph = input("Enter the new restaurant description: ")
                new_record = new_data_entry_process(paragraph, itemId)
                data.append(new_record)
                save_data(data, file_path, backup_path)

                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                index = input("Enter record index to edit: ")
                if index.isdigit() and 0 <= int(index) < len(data):
                    index = int(index)
                    record = data[index]
                    for key in record:
                        new_value = input(f"{key} [{record[key]}]: ")
                        if new_value != '':
                            record[key] = new_value
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5': # DELETE DATA
                index = input("Enter record index to delete: ")
                if index.isdigit() and 0 <= int(index) < len(data):
                    data.pop(int(index))
                    save_data(data, file_path, backup_path)
                    print("✅ Record deleted.")
                else:
                    print("invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")

class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', '1', '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main() # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
