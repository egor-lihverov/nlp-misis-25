import torch
from dataclasses import dataclass
from typing import List, Tuple
from project_config import INPUT_TEXT_HEDGEHOG, INPUT_TEXT_JSON, MODEL, TOKENIZER, DEVICE, EOS_TOKEN_ID

@dataclass
class Candidate:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    score: float

def generate_beam_search(model, tokenizer, input_ids, attention_mask, max_length=50, num_beams=4, length_penalty=1.0, eos_token_id=None,):
    model.eval()
    device = input_ids.device

    unfinished_candidates = []
    finished_candidates = []

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
        top_log_probs, top_indices = torch.topk(log_probs, num_beams)

        for i in range(num_beams):
            new_input_ids = torch.cat([input_ids, top_indices[i].unsqueeze(0).unsqueeze(0)], dim=1)
            new_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
            candidate = Candidate(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                score=top_log_probs[i].item()
            )
            if eos_token_id is not None and top_indices[i].item() == eos_token_id:
                finished_candidates.append(candidate)
            else:
                unfinished_candidates.append(candidate)

    while len(finished_candidates) < num_beams and unfinished_candidates and input_ids.size(1) < max_length:
        new_candidates: List[Tuple[float, Candidate]] = []

        for candidate in unfinished_candidates:
            outputs = model(input_ids=candidate.input_ids, attention_mask=candidate.attention_mask)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, num_beams)

            for i in range(num_beams):
                new_input_ids = torch.cat([candidate.input_ids, top_indices[i].unsqueeze(0).unsqueeze(0)], dim=1)
                new_attention_mask = torch.cat([candidate.attention_mask, torch.ones((1, 1), device=device)], dim=1)
                new_score = candidate.score + top_log_probs[i].item()
                new_candidate = Candidate(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    score=new_score
                )
                new_candidates.append((new_score, new_candidate))

        new_candidates.sort(key=lambda x: x[0], reverse=True)
        unfinished_candidates = []
        for score, candidate in new_candidates[:num_beams]:
            if eos_token_id is not None and candidate.input_ids[0, -1].item() == eos_token_id:
                finished_candidates.append(candidate)
            else:
                unfinished_candidates.append(candidate)

        unfinished_candidates = unfinished_candidates[:num_beams - len(finished_candidates)]

    finished_candidates.extend(unfinished_candidates)


    best_candidate = None
    best_score = float('-inf')
    for candidate in finished_candidates:
        length = candidate.input_ids.size(1)
        adjusted_score = candidate.score / (length ** length_penalty)
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_candidate = candidate

    generated_text = tokenizer.decode(best_candidate.input_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":

    encoding_hedgehog = TOKENIZER(INPUT_TEXT_HEDGEHOG, return_tensors="pt")
    input_ids_hedgehog = encoding_hedgehog["input_ids"].to(DEVICE)
    attention_mask_hedgehog = encoding_hedgehog["attention_mask"].to(DEVICE)

    encoding_json = TOKENIZER(INPUT_TEXT_JSON, return_tensors="pt")
    input_ids_json = encoding_json["input_ids"].to(DEVICE)
    attention_mask_json = encoding_json["attention_mask"].to(DEVICE)

    params = [
        {"num_beams": 1, "length_penalty": 1.0, "label": "num_beams=1, length_penalty=1.0"},
        {"num_beams": 4, "length_penalty": 1.0, "label": "num_beams=4, length_penalty=1.0"},
        {"num_beams": 4, "length_penalty": 0.5, "label": "num_beams=4, length_penalty=0.5"},
        {"num_beams": 4, "length_penalty": 2.0, "label": "num_beams=4, length_penalty=2.0"},
        {"num_beams": 8, "length_penalty": 1.0, "label": "num_beams=8, length_penalty=1.0"},
    ]

    results_hedgehog = []
    results_json = []

    for param in params:
        text_hedgehog = generate_beam_search(
            MODEL, TOKENIZER, input_ids_hedgehog, attention_mask_hedgehog,
            max_length=1000, num_beams=param["num_beams"], length_penalty=param["length_penalty"], eos_token_id=EOS_TOKEN_ID
        )
        results_hedgehog.append({"params": param["label"], "text": text_hedgehog})

        text_json = generate_beam_search(
            MODEL, TOKENIZER, input_ids_json, attention_mask_json,
            max_length=1000, num_beams=param["num_beams"], length_penalty=param["length_penalty"], eos_token_id=EOS_TOKEN_ID
        )
        results_json.append({"params": param["label"], "text": text_json})

    print("=== Generated Hedgehog Stories ===")
    for result in results_hedgehog:
        print(f"\n{result['params']}:\n{result['text']}")

    print("\n=== Generated JSONs ===")
    for result in results_json:
        print(f"\n{result['params']}:\n{result['text']}")