import datasets

openbooqa_main = datasets.load_dataset('openbookqa', 'main', split='train')
openbooqa_additional = datasets.load_dataset('openbookqa', 'additional', split='train')

prompts = []
for i in openbooqa_main:
    prompt = f"""Question: {i['question_stem']}
    A. {i['choices']['text'][0]}
    B. {i['choices']['text'][1]}
    C. {i['choices']['text'][2]}
    D. {i['choices']['text'][3]}
    Answer:{i['answerKey']}"""
    prompts.append(prompt)

for i in openbooqa_additional:
    prompt = f"""Question: {i['question_stem']}
    A. {i['choices']['text'][0]}
    B. {i['choices']['text'][1]}
    C. {i['choices']['text'][2]}
    D. {i['choices']['text'][3]}
    Answer:{i['answerKey']}
    """
    prompts.append(prompt)
