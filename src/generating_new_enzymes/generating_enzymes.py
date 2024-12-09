import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math
import gc

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

    device = torch.device("cuda")
    torch.cuda.empty_cache()

    print('Reading pretrained model and tokenizer')
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    labels=['3.1.1.1', '3.1.1.74', '3.1.1.3', '3.1.1.101', '3.1.1.102',
            '3.5.2.12', '3.5.1.46', '3.5.1.117']

    for label in tqdm(labels):
        print("Generating sequences for label: ", label)
        for i in range(0,10000): 
            tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')
            model = GPT2LMHeadModel.from_pretrained('AI4PD/ZymCTRL').to(device)
    
            sequences = main(label, model, special_tokens, device, tokenizer)
            for key,value in sequences.items():
                for index, val in enumerate(value):
                    fn = open(f"../../results/generated_sequences/{label}_{i}_{index}.fasta", "w")
                    fn.write(f'>{label}_{i}_{index}\t{val[1]}\n{val[0]}')
                    fn.close()

            torch.cuda.empty_cache()
            del tokenizer
            del model
            gc.collect()