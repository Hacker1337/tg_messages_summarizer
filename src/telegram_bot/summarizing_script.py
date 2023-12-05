import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import yaml
import json
import logging


with open("sum_config.yaml") as file:
    config = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
model = AutoModelForSeq2SeqLM.from_pretrained(config["checkpoint"])

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def convert_model_output_to_text(model_output, names):
    '''
    # usage example
    model_output = "#Person1# танцует с Брайаном на вечеринке по случаю дня рождения Брайана. Брайан считает, что #Person1# выглядит великолепно и пользуется популярностью."
    names = ["Алексей", "Мария", "Иван"]

    result_text = replace_patterns(model_output, names)
    print(result_text)
    # output: Алексей танцует с Брайаном на вечеринке по случаю дня рождения Брайана. Брайан считает, что Алексей выглядит великолепно и пользуется популярностью.
    '''
    pattern = re.compile(r'#(?:Person|Человек)(\d+)#')

    def replace_match(match):
        index = int(match.group(1)) - 1
        if 0 <= index < len(names):
            return names[index]
        else:
            return match.group(0)
    result = pattern.sub(replace_match, model_output)
    return result

def form_prompt(messages):
    prompt_parts = []

    id2user = {}

    for message in messages:
        if not (hasattr(message, 'forward_from') and message.forward_from):
            logging.warn("message has no forwarded field, while only such messages shoud be provided. Skip this message. Message is:\n"+str(message))
            continue
        if not hasattr(message.forward_from, "id"):
            logging.warn("forwarded field has no id. Skip this message. Message is:\n" + str(message) + "\nForward field is:\n" + str(message.forward_from))
            continue

        if message.forward_from.id not in id2user:
            number = len(id2user) + 1
            user_name = message.forward_from.full_name
            id2user[message.forward_from.id] = [user_name, number]
        else:
            number = id2user[message.forward_from.id][1]

        prompt_parts.append(f"#Человек{number}#: {message.text}")  # like #Человек2#:

    prompt = "\n".join(prompt_parts)
    person_number_pairs = sorted(id2user.values(), key=lambda x: x[1])
    persons_list = [x[0] for x in person_number_pairs]

    return prompt, persons_list

def summarize_messages(messages):
    prompt, persons_list = form_prompt(messages)
    logging.debug("Prompt:\n" + prompt)
    model_output = summarizer(prompt)[0]["summary_text"]
    result_text = convert_model_output_to_text(model_output, persons_list)
    logging.info("Prompt:\n" + prompt + "\nResult:\n" + result_text)

    data = {
        "prompt": prompt,
        "model_output": model_output,   # store data in anonymous way
    }
    with open("logs/work_results.json", 'a', encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

    return result_text