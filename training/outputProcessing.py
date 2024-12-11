import json

# Load the JSON file
with open("output_responses.json", "r") as file:
    data = json.load(file)

# Create an empty list to hold the reformatted data
formatted_data = []

# Iterate over the original JSON data
for idx, (question, content) in enumerate(data.items()):
    # Extract the correct answer
    correct_answer = content["correct_response"]
    
    # Extract the generated response
    generated_response = content["generated_response"]

    # Add the transformed entry to the formatted data
    formatted_data.append({
        "id": str(idx + 1),
        "correct_response": correct_answer,
        "generated_response": generated_response
    })

# Save the formatted data to a new JSON file
with open("output.json", "w") as file:
    json.dump(formatted_data, file, indent=4)

print("Transformation complete! Data saved to output.json.")
