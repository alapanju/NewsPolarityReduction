#This is the code for adversarial attack based contextual perturbation in sentence for sentiment modification

import os
import sys
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack import Attack, Attacker, AttackArgs

# from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.overlap import MaxWordsPerturbed
# from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from modified_textattack.constraints import BERT, Entailment

# from textattack.goal_functions import TargetedClassification
from modified_textattack.goal_functions import ModifiedTargetedClassification

# from textattack.search_methods import GeneticAlgorithm
from modified_textattack.search_methods import ModifiedBeamSearch

from textattack.transformations import WordSwapMaskedLM, WordInsertionMaskedLM, WordMergeMaskedLM, WordDeletion
from textattack.transformations import CompositeTransformation

random_seed = 1

data_folder = 'datasets'
output_folder = sys.argv[1]
gpu_id = int(sys.argv[2])
use_swap = sys.argv[3]
use_swap = True if use_swap=="1" else False
use_ins = sys.argv[4]
use_ins = True if use_ins=="1" else False
use_del = sys.argv[5]
use_del = True if use_del=="1" else False
use_ent = sys.argv[6]
use_ent = True if use_ent=="1" else False
ent_rate = float(sys.argv[7])
target_class = int(sys.argv[8])
beam_size = int(sys.argv[9])

torch.cuda.set_device(gpu_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

sent_model_folder = '/home/alapan/media_bias/temp5/sentiment_900'

model = AutoModelForSequenceClassification.from_pretrained(sent_model_folder, local_files_only=True).to(device)

tokenizer = AutoTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

train_file = os.path.join(data_folder, "all_sentiment_train.csv")
dev_file = os.path.join(data_folder, "all_sentiment_dev.csv")
train_output_file = os.path.join(output_folder, "all_sentiment_train.csv")
dev_output_file = os.path.join(output_folder, "all_sentiment_dev.csv")

output_columns = ["input","aspect","output"]

def style_transfer(input_file, output_file):
    output_data = pd.DataFrame(columns= output_columns)
    if os.path.exists(output_file):
        output_data = pd.read_csv(output_file)


    df = pd.read_csv(input_file)
    rows = len(output_data)

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    shared_masked_lm = AutoModelForCausalLM.from_pretrained("distilroberta-base")
    shared_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    composite_list = []
    if use_swap:
        composite_list.append(WordSwapMaskedLM(
            method="bae",
            masked_language_model=shared_masked_lm,
            tokenizer=shared_tokenizer,
            max_candidates=10,
            min_confidence=5e-4,
        ))
    if use_ins:
        composite_list.append(WordInsertionMaskedLM(
            masked_language_model=shared_masked_lm,
            tokenizer=shared_tokenizer,
            max_candidates=10,
            min_confidence=5e-4,
        ))
    if use_del:
        composite_list.append(WordDeletion())

    transformation = CompositeTransformation(composite_list)

    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(InputColumnModification(['sentence','hypothesis'],['hypothesis']))
    # constraints.append(UniversalSentenceEncoder(
    #     threshold=0.9,
    #     metric="cosine",
    #     compare_against_original=True,
    #     # window_size=15,
    #     # skip_text_shorter_than_window=True,
    # ))
    # constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
    constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=0))
    # constraints.append(MaxWordsPerturbed(max_num_words=1))
    constraints.append(BERT(model_name="stsb-distilbert-base",threshold=0.95,metric="cosine"))
    if use_ent:
        constraints.append(Entailment(threshold=ent_rate))

    goal_function = ModifiedTargetedClassification(model_wrapper, target_class=target_class, maximizable=True)

    search_method = ModifiedBeamSearch(beam_width=beam_size)

    attack = Attack(goal_function, constraints, transformation, search_method)
    attack_args = AttackArgs(num_examples = -1, random_seed = random_seed, disable_stdout=True, silent=True)

    for index, row in tqdm(df.iterrows()):
        if index<rows:
            continue

        torch.cuda.empty_cache()

        data = [((row['sentence'][9:],row['hypothesis']),row['label'])]

        dataset = Dataset(data, input_columns=("sentence","hypothesis"), label_names=['positive','neutral','negative'])

        attacker = Attacker(attack, dataset, attack_args)
        attack_results = attacker.attack_dataset()


        for attack_result in attack_results:
            sentence = attack_result.original_result.attacked_text.text.split("\n")[0]
            aspect = attack_result.original_result.attacked_text.text.split("\n")[1][:-20]
            output = attack_result.perturbed_result.attacked_text.text.split("\n")[0]

            output_row = pd.Series([sentence, aspect, output], index=output_columns)
            output_data = output_data.append(output_row, ignore_index=True)
            output_data.to_csv(output_file, index=False)

        print(f"Done {index+1} out of {len(df)}")


style_transfer(train_file, train_output_file)

style_transfer(dev_file, dev_output_file)
