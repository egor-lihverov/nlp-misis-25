import torch
from project_config import INPUT_TEXT_HEDGEHOG, INPUT_TEXT_JSON, MODEL, TOKENIZER, DEVICE, EOS_TOKEN_ID

def generate_nucleus_sampling(model, tokenizer, input_ids, attention_mask, max_length=50, temperature=1.0, top_p=0.9, eos_token_id=None):
    model.eval()
    generated_ids = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            outputs = model(input_ids=generated_ids, attention_mask=generated_attention_mask)
            logits = outputs.logits
            next_token_logits = logits[0, -1] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            mask = cumulative_probs <= top_p
            if not mask.any():
                mask[0] = True

            filtered_probs = sorted_probs[mask]
            filtered_indices = sorted_indices[mask]

            filtered_probs = filtered_probs / filtered_probs.sum()

            next_token_idx = torch.multinomial(filtered_probs, num_samples=1)
            next_token_id = filtered_indices[next_token_idx].unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token_id)], dim=1
            )

            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":

    params = [
        {"temperature": 1.0, "top_p": 0.9, "label": "temperature=1.0, top_p=0.9"},
        {"temperature": 1.0, "top_p": 0.15, "label": "temperature=1.0, top_p=0.15"},
        {"temperature": 0.5, "top_p": 0.9, "label": "temperature=0.5, top_p=0.9"},
        {"temperature": 0.5, "top_p": 0.15, "label": "temperature=0.5, top_p=0.15"},
    ]

    encoding_hedgehog = TOKENIZER(INPUT_TEXT_HEDGEHOG, return_tensors="pt")
    input_ids_hedgehog = encoding_hedgehog["input_ids"].to(DEVICE)
    attention_mask_hedgehog = encoding_hedgehog["attention_mask"].to(DEVICE)

    encoding_json = TOKENIZER(INPUT_TEXT_JSON, return_tensors="pt")
    input_ids_json = encoding_json["input_ids"].to(DEVICE)
    attention_mask_json = encoding_json["attention_mask"].to(DEVICE)

    results_hedgehog = []
    results_json = []

    for param in params:

        text_hedgehog = generate_nucleus_sampling(
            MODEL, TOKENIZER, input_ids_hedgehog, attention_mask_hedgehog,
            max_length=1000, temperature=param["temperature"], top_p=param["top_p"], eos_token_id=EOS_TOKEN_ID
        )
        results_hedgehog.append({"params": param["label"], "text": text_hedgehog})


        text_json = generate_nucleus_sampling(
            MODEL, TOKENIZER, input_ids_json, attention_mask_json,
            max_length=1000, temperature=param["temperature"], top_p=param["top_p"], eos_token_id=EOS_TOKEN_ID
        )
        results_json.append({"params": param["label"], "text": text_json})


    print("=== Generated Hedgehog Stories ===")
    for result in results_hedgehog:
        print(f"\n{result['params']}:\n{result['text']}")

    print("\n=== Generated JSONs ===")
    for result in results_json:
        print(f"\n{result['params']}:\n{result['text']}")