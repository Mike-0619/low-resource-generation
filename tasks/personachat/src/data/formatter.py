# src/data/formatter.py
import random

def build_personachat_messages(example, max_history_turns=10):
    """
    Build the prompt messages (system + history) for one PersonaChat example.

    Single source of truth shared by training (formatter) and evaluation
    (evaluator) — do not duplicate this logic elsewhere. The final assistant
    turn is NOT included: callers append the target reply (train) or the
    candidate being scored (eval).
    """
    persona_text = " ".join(example['personality'])
    messages = [{"role": "system", "content": f"Persona: {persona_text}"}]

    recent_history = example['history'][-max_history_turns:]

    # The last history turn is always the user, so the first role follows from length
    is_user_turn = (len(recent_history) % 2 != 0)
    for turn in recent_history:
        messages.append({"role": "user" if is_user_turn else "assistant", "content": turn})
        is_user_turn = not is_user_turn

    return messages


def format_sft_personachat(example):
    """
    Format data for SFT using the Universal Messages format.
    Returns a dictionary with a single 'messages' list.
    """
    messages = build_personachat_messages(example)

    # Append the target reply as the final assistant message
    target_reply = example['candidates'][-1]
    messages.append({"role": "assistant", "content": target_reply})

    # Return pure messages, completely decoupled from the tokenizer
    return {"messages": messages}

def format_dpo_personachat(example):
    """
    Format data for DPO: Separates the history (prompt) from the answers (chosen vs rejected).
    """
    prompt_messages = build_personachat_messages(example)

    target_reply = example['candidates'][-1]
    
    # Pick a random distractor as the rejected reply
    distractors = example['candidates'][:-1]
    rejected_reply = random.choice(distractors)

    return {
        "prompt": prompt_messages, 
        "chosen": [{"role": "assistant", "content": target_reply}], 
        "rejected": [{"role": "assistant", "content": rejected_reply}]
    }