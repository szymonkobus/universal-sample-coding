import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def get_input(tokenizer, dataset, data_col, max_len, device):
    dataset = load_dataset(dataset)
    data = dataset['train'][data_col]
    token_out = tokenizer(data, return_tensors='pt', padding=True, 
                          truncation=True, max_length=max_len)
    input_ids = token_out.input_ids.to(device)
    mask = input_ids != tokenizer.pad_token_id
    return input_ids, mask

def parse_input(in_ids, in_mask, batch, epoch):
    L = in_ids.shape[0]
    start = epoch * batch
    end = start + batch
    while start >= L:
        start -= L
        end -= L
    if end > L:
        in_ids_cat = torch.cat((in_ids[start:L], in_ids[:end-L]))
        in_mask_cat = torch.cat((in_mask[start:L], in_mask[:end-L]))
        return in_ids_cat, in_mask_cat
    return in_ids[start:end], in_mask[start:end]

def get_prob(model, in_ids, dev, dev_com):
    with torch.no_grad():
        outputs = model(in_ids.to(dev))
        logits_e = outputs.logits[:,-1,:]
        prob = torch.nn.functional.softmax(logits_e, dim=-1)
        log_p = torch.log2(prob + 1e-7) # avoid log(0) for float16
    return prob.to(dev_com), log_p.to(dev_com)

def get_prob_marked(model, in_ids, dev, dev_com, i):
    return get_prob(model, in_ids, dev, dev_com), i

def get_probs(models, in_gen, devs, dev_com):
    probs, log_ps= [],[]
    for i, model in enumerate(models):
        prob, log_p = get_prob(model, in_gen, devs[i], dev_com)
        probs.append(prob.to(dev_com))
        log_ps.append(log_p.to(dev_com))
    return probs, log_ps

def get_probs_parallel(models, in_gen, devs, dev_com):
    probs, log_ps= [None]*len(models), [None]*len(models)
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, model in reversed(list(enumerate(models))):
            in_gen_i = in_gen.to(devs[i]).detach().clone()
            job = executor.submit(get_prob_marked, model, in_gen_i, devs[i], dev_com, i)
            futures.append(job)
        complete_futures, incomplete_futures = wait(futures)
        for f in complete_futures:
            (prob, log_p), i = f.result()
            probs[i] = prob
            log_ps[i] = log_p
    return probs, log_ps

def cross_entropy(p, log2_q):
    return -torch.sum(p*log2_q, dim=-1)

def entropy(p, log2_p):
    return cross_entropy(p, log2_p)

def kl_div(p, log2_p, log2_q):
    return cross_entropy(p, log2_q) - entropy(p, log2_p)

def main(conf):
    batch = conf['batch']
    n_tokens = conf['n_tokens']
    n_epochs = conf['n_epochs']
    model_names = conf['model_names']
    devs = conf['devs']
    main_model_i = conf['main_model_i']
    dev_com = conf['dev_com']

    logging.info('Parsing input.')
    tokenizer = AutoTokenizer.from_pretrained(model_names[main_model_i])
    in_ids_all, in_mask_all = get_input(tokenizer, conf['dataset_name'], 
                                        conf['data_col'], conf['max_len'], dev_com)
    models = []
    for model_name, dev, dtype in zip(model_names, devs, conf['dtypes']):
        logging.info('Loading model {}.'.format(model_name))
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(dev)
        model.eval()
        models.append(model)

    size = (len(models), len(models), n_epochs, batch, n_tokens)
    divergence = torch.zeros(size, dtype=torch.float32, device=dev_com)
    token_sum_prob = torch.zeros((tokenizer.vocab_size+7,), device=dev_com)
    
    for epoch in tqdm(range(n_epochs), desc='epoch', position=0, ncols=80):
        in_ids, in_mask = parse_input(in_ids_all, in_mask_all, batch, epoch)
        in_gen = torch.zeros((batch,2), dtype=torch.long, device=dev_com)
        for i in tqdm(range(1, n_tokens+1), desc='token', position=1, leave=False, ncols=80):
            fix_i = min(i+1, in_ids.shape[1])
            in_gen[:,:fix_i] = in_ids[:,:fix_i]*in_mask[:,:fix_i] \
                                + in_gen[:,:fix_i]*(~in_mask[:,:fix_i])
            
            # probs, log_ps= get_probs(models, in_gen, devs, dev_com) 
            probs, log_ps= get_probs_parallel(models, in_gen, devs, dev_com)

            lead_prob = probs[main_model_i]
            next_gen = torch.multinomial(lead_prob, num_samples=1)
            in_gen = torch.cat((in_gen, next_gen), dim=1)

            # CALCULATING METRICS 
            for mod_i in range(len(models)):
                for mod_j in range(len(models)):
                    if mod_i == mod_j:
                        metric = entropy(probs[mod_i], log_ps[mod_i])
                    else:
                        metric = kl_div(probs[mod_i], log_ps[mod_i], log_ps[mod_j])
                    divergence[mod_i, mod_j, epoch, :, i-1] = metric
            torch.save(divergence.to('cpu'), '{}/divergence.pt'.format(conf['dir']))
            token_sum_prob += torch.sum(lead_prob, dim=0)
            torch.save(token_sum_prob.to('cpu'), '{}/token_cum_probs.pt'.format(conf['dir']))

    # LOGGING RESULTS
    for mod_i in range(len(models)):
        for mod_j in range(len(models)):
            div = divergence[mod_i,mod_j].sum(dim=-1).mean()
            logging.info('D(P_{}||P_{}) = {}'.format(mod_i, mod_j, div))

if __name__ == "__main__":
    # CONFIGURATION
    models_idxs = [0, 1, 2, 3, 4, 5]
    main_model_i = len(models_idxs)-1
    all_models = ['facebook/opt-{}'.format(x) for x in 
                  ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b']]
    common = 'cuda:0'
    all_devs = ['cuda:2', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6']
    all_dtypes = [torch.float32]*5 + [torch.float16]*3
    dir_path = 'results/{}'.format(get_timestamp())

    conf = {
        'batch': 100,
        'n_tokens': 128 + 10,
        'n_epochs': 500,
        'model_names': [all_models[i] for i in models_idxs],
        'devs': [all_devs[i] for i in models_idxs],
        'dtypes': [all_dtypes[i] for i in models_idxs],
        'main_model_i': main_model_i,
        'dev_com': common,
        'dataset_name': 'openbookqa', 
        'data_col': 'question_stem',
        'max_len': 10,
        'dir': dir_path,
        'seed': 20230809,
    }

    # LOGGING AND SAVING RESULTS
    os.makedirs(dir_path, exist_ok=True)
    with open('{}/config.txt'.format(dir_path), 'w') as f:
        f.write(str(conf))
    frmt = '%(asctime)s - %(message)s'
    datefmt = '%y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=frmt, datefmt=datefmt)
    torch.manual_seed(conf['seed'])
    
    open('{}/running.log'.format(dir_path), 'w').close()
    main(conf)
    os.remove('{}/running.log'.format(dir_path))