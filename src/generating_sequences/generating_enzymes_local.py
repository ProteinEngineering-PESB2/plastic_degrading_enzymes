import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import pandas as pd
import math
import gc
import sys

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model,special_tokens,device,tokenizer):
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=5) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    ppls.sort(key=lambda i:i[1]) # duplicated sequences?

    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':

    label = sys.argv[1]

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    print('Reading pretrained model and tokenizer')
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    #labels=['3.1.1.74', '3.1.1.3', '3.1.1.101', '3.1.1.102',
    #        '3.5.2.12', '3.5.1.46', '3.5.1.117']

    print("Generating sequences for label: ", label)
    matrix_data = []
    for i in range(0,2000): 
        tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')
        model = GPT2LMHeadModel.from_pretrained('AI4PD/ZymCTRL').to(device)

        sequences = main(label, model, special_tokens, device, tokenizer)
        for key,value in sequences.items():
            for index, val in enumerate(value):
                row = [val[0], val[1]]
                matrix_data.append(row)
                
        torch.cuda.empty_cache()
        del tokenizer
        del model
        gc.collect()

    df_export = pd.DataFrame(data=matrix_data, columns=["sequence", "perplexity"])
    df_export.to_csv(f"/scratch/global_1/dmedina/results_generating_enzymes/generated_sequences/{label}.csv", index=False)