import json
import openai
import evaluate
import os
import pandas as pd
from pprint import pprint

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("KEYGPT"),
    organization="org-CijVCVQH1zqycayGlJnTKgJH",
    project="proj_ail4ANzX7UrB4qeFXRLgatPh",
)

# Load BLEU and TER metrics for evaluation
bleu = evaluate.load("sacrebleu")
ter = evaluate.load("ter")

# Path to the JSONL file containing test data
test_data_file = "val_espanol_mixtec.jsonl"

# Read test data from the JSONL file
test_data = []
with open(test_data_file, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))  # Load each line as a JSON object

# Lists to store model translations and reference translations
predictions = []
references = []

# Generate translations using the fine-tuned model
for entry in test_data:
    messages = entry["messages"]

    # Extract source sentence (input text) and expected reference (ground truth)
    source_text = messages[1]["content"]  # User input (Spanish or Mixtec)
    target_text = messages[2]["content"]  # Expected translation (assistant response)

    try:
        # API call to OpenAI fine-tuned model for translation
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:gpt-4o-mini-2024-07-18-gpt-spanish-mixtec-05-05:B892Bp3r",
            messages=[
                {
                    "role": "system",
                    "content": "You are a translator between Spanish and Mixtec. "
                               "If the input is in Spanish, translate it to Mixtec; "
                               "if the input is in Mixtec, translate it to Spanish."
                },
                {"role": "user", "content": source_text}
            ],
            # max_tokens=128000  # Optional token limit (commented out)
        )

        # Extract the model's response (translation)
        translated_text = completion.choices[0].message.content.strip()
        predictions.append(translated_text)  # Store model-generated translation
        references.append([target_text])  # Store ground truth translation (nested list format)

    except openai.APIError as e:
        print(f"❌ API Error: {e}")
    except openai.AuthenticationError as e:
        print(f"🔑 Authentication Error: {e}")
    except openai.RateLimitError as e:
        print(f"⏳ Rate Limit Reached: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")
        print("🔍 API Response:", completion if 'completion' in locals() else "Not available")

# Compute BLEU and TER metrics
bleu_score = bleu.compute(predictions=predictions, references=references)
ter_score = ter.compute(predictions=predictions, references=references)

# Display evaluation results
print(f"BLEU: {bleu_score['score']:.4f}")
print(f"TER: {ter_score['score']:.4f}")

# Show a few sample translations for reference
for i in range(5):  # Display 5 random examples
    print("\n🔹 Input:", test_data[i]["messages"][1]["content"])
    print("✅ Reference:", references[i][0])
    print("⚡ Model Translation:", predictions[i])

# Format the results for saving
results = f"BLEU: {bleu_score['score']:.4f}\nTER: {ter_score['score']:.4f}"

# Print results to console
print(results)

# Save evaluation results to a text file
with open("metrics_results_spanish_mixtec.txt", "w", encoding="utf-8") as file:
    file.write(results)

print("Metrics saved in 'metrics_results_spanish_mixtec.txt'.")