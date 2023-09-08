import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from tqdm import tqdm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to(torch.device('cuda'))

bigbench_task_list = ['abstract_narrative_understanding', 'anachronisms', 'analogical_similarity',
                      'analytic_entailment', 'arithmetic',
                      'ascii_word_recognition', 'authorship_verification', 'auto_categorization', 'auto_debugging',
                      'bbq_lite_json',
                      'bridging_anaphora_resolution_barqa', 'causal_judgment', 'cause_and_effect', 'checkmate_in_one',
                      'chess_state_tracking', 'chinese_remainder_theorem',
                      'code_line_description',
                      'codenames', 'color', 'common_morpheme', 'conceptual_combinations', 'conlang_translation',
                      'contextual_parametric_knowledge_conflicts', 'crash_blossom', 'crass_ai', 'cryobiology_spanish',
                      'cryptonite',
                      'cs_algorithms', 'dark_humor_detection', 'date_understanding', 'disambiguation_qa',
                      'discourse_marker_prediction',
                      'disfl_qa', 'dyck_languages', 'elementary_math_qa', 'emoji_movie', 'emojis_emotion_prediction',
                      'empirical_judgments', 'english_proverbs', 'english_russian_proverbs', 'entailed_polarity',
                      'entailed_polarity_hindi', 'epistemic_reasoning', 'evaluating_information_essentiality',
                      'fact_checker',
                      'fantasy_reasoning', 'few_shot_nlg', 'figure_of_speech_detection',
                      'formal_fallacies_syllogisms_negation', 'gem',
                      'gender_inclusive_sentences_german', 'general_knowledge', 'geometric_shapes', 'goal_step_wikihow',
                      'gre_reading_comprehension', 'hhh_alignment', 'hindi_question_answering', 'hindu_knowledge',
                      'hinglish_toxicity',
                      'human_organs_senses', 'hyperbaton', 'identify_math_theorems', 'identify_odd_metaphor',
                      'implicatures',
                      'implicit_relations', 'intent_recognition', 'international_phonetic_alphabet_nli',
                      'international_phonetic_alphabet_transliterate', 'intersect_geometry', 'irony_identification',
                      'kanji_ascii',
                      'kannada', 'key_value_maps', 'known_unknowns', 'language_games', 'language_identification',
                      'linguistic_mappings',
                      'linguistics_puzzles', 'list_functions', 'logic_grid_puzzle', 'logical_args', 'logical_deduction',
                      'logical_fallacy_detection', 'logical_sequence', 'mathematical_induction', 'matrixshapes',
                      'metaphor_boolean',
                      'metaphor_understanding', 'minute_mysteries_qa', 'misconceptions', 'misconceptions_russian',
                      'mnist_ascii',
                      'modified_arithmetic', 'moral_permissibility', 'movie_dialog_same_or_different',
                      'movie_recommendation',
                      'mult_data_wrangling', 'multiemo', 'natural_instructions', 'navigate', 'nonsense_words_grammar',
                      'novel_concepts',
                      'object_counting', 'odd_one_out', 'operators', 'paragraph_segmentation', 'parsinlu_qa',
                      'parsinlu_reading_comprehension', 'penguins_in_a_table', 'periodic_elements', 'persian_idioms',
                      'phrase_relatedness', 'physical_intuition', 'physics', 'physics_questions',
                      'play_dialog_same_or_different',
                      'polish_sequence_labeling', 'presuppositions_as_nli', 'qa_wikidata', 'question_selection',
                      'real_or_fake_text',
                      'reasoning_about_colored_objects', 'repeat_copy_logic', 'rephrase', 'riddle_sense', 'ruin_names',
                      'salient_translation_error_detection', 'scientific_press_release',
                      'semantic_parsing_in_context_sparc',
                      'semantic_parsing_spider', 'sentence_ambiguity', 'similarities_abstraction',
                      'simp_turing_concept',
                      'simple_arithmetic_json', 'simple_arithmetic_json_multiple_choice',
                      'simple_arithmetic_json_subtasks',
                      'simple_arithmetic_multiple_targets_json', 'simple_ethical_questions', 'simple_text_editing',
                      'snarks',
                      'social_iqa', 'social_support', 'sports_understanding', 'strange_stories', 'strategyqa',
                      'sufficient_information',
                      'suicide_risk', 'swahili_english_proverbs', 'swedish_to_german_proverbs', 'symbol_interpretation',
                      'temporal_sequences', 'tense', 'timedial', 'topical_chat', 'tracking_shuffled_objects',
                      'understanding_fables',
                      'undo_permutation', 'unit_conversion', 'unit_interpretation', 'unnatural_in_context_learning',
                      'vitaminc_fact_verification', 'what_is_the_tao', 'which_wiki_edit', 'winowhy', 'word_sorting',
                      'word_unscrambling']


def create_big_bench_split(split_name):
    data = load_dataset("bigbench", split_name)['train']
    data = data.shuffle(42)
    data = data.select(range(int(len(data) * 0.01)))
    return data


dataset = Parallel(n_jobs=16)(delayed(create_big_bench_split)(task_name) for task_name in tqdm(bigbench_task_list))

dataset = concatenate_datasets(dataset)
dataset.save_to_disk("/home/mithil/PycharmProjects/NeuripsLLMEfficiency/data/bigbench_train_small")


def return_prompt(example):
    prompt = f"""The following are multiple choice questions . Please choose the correct answer from the four choices
    Question: {example['inputs']}
    Options: {example['multiple_choice_targets']}
    targets. {example['targets']}"""
    return {'output_text': prompt}
