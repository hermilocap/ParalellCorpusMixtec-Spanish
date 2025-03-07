import json
import openai
import evaluate
import os
import pandas as pd
from pprint import pprint

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("KEYGPT"),  # Retrieve API key from environment variable
    organization="org-CijVCVQH1zqycayGlJnTKgJH",  # Organization ID
    project="proj_ail4ANzX7UrB4qeFXRLgatPh",  # Project ID
)

# Load BLEU and TER evaluation metrics
bleu = evaluate.load("sacrebleu")  
ter = evaluate.load("ter")  

# Path to the JSONL file containing test data
test_data_file = "val_espanol_mixtec.jsonl"

# Read test data from the JSONL file
test_data = []
with open(test_data_file, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))  # Load each JSON line as a dictionary

# Prepare lists for model translations and reference translations
predictions = []
references = []

# Iterate over test samples to translate them using the fine-tuned model
for entry in test_data:
    messages = entry["messages"]
    
    # Extract the source sentence and the correct reference translation
    source_text = messages[1]["content"]  # The input text (either Spanish or Mixtec)
    target_text = messages[2]["content"]  # The expected translation (assistant's response)

    try:
        # Call the OpenAI API to get the model's translation
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:gpt-4o-mini-2024-07-18-gpt-spanish-mixtec-05-05:B892Bp3r",
            messages=[
                {"role": "system", "content": "You are a translator between Spanish and Mixtec. If the input text is in Spanish, translate it to Mixtec; if it is in Mixtec, translate it to Spanish."},
                {"role": "user", "content": source_text}
            ],
            #max_tokens=128000  # Uncomment if needed to specify a token limit
        )

        # Extract the model's response
        translated_text = completion.choices[0].message.content.strip()
        predictions.append(translated_text)  # Store the model's translation
        references.append([target_text])  # Store the reference translation (in a nested list)

    # Handle API errors
    except openai.APIError as e:
        print(f"❌ API Error: {e}")
    except openai.AuthenticationError as e:
        print(f"🔑 Authentication Error: {e}")
    except openai.RateLimitError as e:
        print(f"⏳ Rate Limit Reached: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")
        print("🔍 API Response:", completion if 'completion' in locals() else "Not available")

# Compute BLEU and TER scores for evaluation
bleu_score = bleu.compute(predictions=predictions, references=references)
ter_score = ter.compute(predictions=predictions, references=references)

# Display the evaluation results
print(f"BLEU: {bleu_score['score']:.4f}")
print(f"TER: {ter_score['score']:.4f}")

# Display a few sample predictions
for i in range(5):  # Show 5 random examples
    print("\n🔹 Input:", test_data[i]["messages"][1]["content"])
    print("✅ Reference:", references[i][0])
    print("⚡ Model Translation:", predictions[i])

# Format the evaluation results
results = f"BLEU: {bleu_score['score']:.4f}\nTER: {ter_score['score']:.4f}"

# Print results to the console
print(results)

# Save evaluation results to a text file
with open("metrics_results_mixtec_spanish.txt", "w", encoding="utf-8") as file:
    file.write(results)

print("Metrics saved in 'metrics_results_mixtec_spanish.txt'.")