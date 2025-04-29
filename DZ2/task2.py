import torch
from tqdm import tqdm
from project_config import INPUT_TEXT_HEDGEHOG, INPUT_TEXT_JSON, MODEL, TOKENIZER, DEVICE, EOS_TOKEN_ID

def generate_sampling(
    model, tokenizer, input_ids, attention_mask, max_length=1000, eos_token_id=None,):
    model.eval()
    generated_ids = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            outputs = model(
                input_ids=generated_ids, attention_mask=generated_attention_mask
            )
            logits = outputs.logits
            next_token_logits = logits[0, -1]

            next_token_id = (
                torch.multinomial(
                    input=torch.softmax(next_token_logits, dim=0), num_samples=1
                )
                .unsqueeze(0)
            )
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token_id)], dim=1
            )
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    sampling_texts = {
        "story": [],
        "json" : [],
    }
    for source in ["story", "json"]:
    
        encoding = TOKENIZER(
            INPUT_TEXT_JSON if source == 'json' else INPUT_TEXT_HEDGEHOG, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        
        for attempt in tqdm(range(2)):
            generated_text = generate_sampling(
                model=MODEL,
                tokenizer=TOKENIZER,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1000,
                eos_token_id=EOS_TOKEN_ID,
            )
            sampling_texts[source].append(generated_text)
    
    print("=== Generated Hedgehog Stories ===")
       
    for text in sampling_texts["story"]:
        print(text)
    
    print("\n=== Generated JSONs ===")
    
    for text in sampling_texts["json"]:
        print(text)