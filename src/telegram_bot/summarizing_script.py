import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import yaml

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
    print(model_output)
    print()
    print(names)
    result = pattern.sub(replace_match, model_output)
    return result

def form_prompt(messages):
    prompt_parts = []

    id2user = {}

    for message in messages:
        if message.forward_from.id not in id2user:
            number = len(id2user) + 1
            user_name = message.forward_from.full_name
            id2user[message.forward_from.id] = [user_name, number]
        else:
            number = id2user[message.from_user.id][1]

        prompt_parts.append(f"#Человек{number}#: {message.text}")  # like #Человек2#:

    prompt = "\n".join(prompt_parts)
    person_number_pairs = sorted(id2user.values(), key=lambda x: x[1])
    persons_list = [x[0] for x in person_number_pairs]

    return prompt, persons_list

def summarize_messages(messages):
    prompt, persons_list = form_prompt(messages)
    print("Prompt:\n", prompt)
    model_output = summarizer(prompt)
    result_text = convert_model_output_to_text(model_output, persons_list)
    print("Result:\n", result_text)
    return result_text