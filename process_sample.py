import logging
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import torch
from tqdm import tqdm


def read_configs(dir, config_require, dev):
    divs = []
    for subdir in os.listdir(dir):
        subdir_path = '{}/{}'.format(dir, subdir)
        config_path = '{}/config.txt'.format(subdir_path)
        running_path = '{}/running.log'.format(subdir_path)
        if os.path.exists(config_path) and not os.path.exists(running_path):
            with open(config_path, 'r') as f:
                config_dict = eval(f.read())
            if valid_config(config_dict, config_require):
                div_path = '{}/divergence.pt'.format(subdir_path)
                if os.path.exists(config_path):
                    divergence = torch.load(div_path).to(dev)
                    divs.append(divergence)
                    logging.info('Read divergence from {}'.format(div_path))
    divs = [d.reshape((d.shape[:2]+(-1,)+d.shape[-1:])) for d in divs]
    return divs

def valid_config(config, config_require):
    # check if config contains all required keys and values
    for k, v in config_require.items():
        if k not in config:
            return False
        if isinstance(v, list):
            if len(config[k])!=len(v) or any(a!=b for a,b in zip(config[k], v)):
                return False
        else:
            if config[k] != v:
                return False
    return True

def coding_cost(x):
    # upper bound of coding cost: H(x) + 1
    return x + 1

def sample_cost(x):
    # upper bound on sample communication: D(Q||P) + log2(D(P||Q)+1) + 5
    return x + torch.log2(x+1) + 5

def calc_metrics(divs, group_size, skip, n_models, main_model_i):
    # calculate metrics for each grouping size of samples
    means = [[] for _ in range(n_models)]
    stds = [[] for _ in range(n_models)]
    medians = [[] for _ in range(n_models)]
    q25s = [[] for _ in range(n_models)]
    q75s = [[] for _ in range(n_models)]
    low_means = [0.0] * n_models

    for group_size in tqdm(group_sizes):
        divs_grouped = []
        for div in divs:
            div = div[:, :, :, skip:]
            n_tokens = (div.shape[-1]//group_size)*group_size
            div = div[:,:,:,:n_tokens].reshape((*div.shape[:2],-1,group_size))
            divs_grouped.append(div.sum(dim=-1))
        div_join = torch.cat(divs_grouped, dim=2)

        for i in range(n_models):
            div_i = div_join[main_model_i, i]
            if i == main_model_i:
                div_i = coding_cost(div_i)
            else:
                div_i = sample_cost(div_i)
            div_i = div_i / group_size
            var, mean = torch.var_mean(div_i)
            means[i].append(mean.item())
            stds[i].append(torch.sqrt(var).item())
            medians[i].append(torch.median(div_i).item())
            q25s[i].append(torch.quantile(div_i, 0.25).item())
            q75s[i].append(torch.quantile(div_i, 0.75).item())

    for i in range(n_models):
        flat_divs = [div[main_model_i, i].reshape(-1) for div in divs]
        low_mean = torch.cat(flat_divs).mean()
        low_means[i]= low_mean.item()

    return means, stds, medians, q25s, q75s, low_means

def plot(plot_path, group_sizes, model_names, main_model_i, metrics):
    means, stds, medians, q25s, q75s, low_means = metrics
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['figure.figsize'] = [6.4, 6.4 * 1.1]
    fig = plt.figure()
    plt.xlabel('group size')
    plt.ylabel('cost per token')
    label = ['sample 125M', 'sample 350M', 'sample 1.3B', 'sample 2.7B', 
             'sample 6.7B', 'code 13B']
    offset = 3
    low_mean_x = 256
    color_coding = plt.colormaps["Oranges"]
    colors_samples = plt.colormaps["Blues"]
    marked_models = list(enumerate(model_names))     
    for i, model in marked_models[-1:] + marked_models[:-1]:
        if i == main_model_i:
            color = color_coding(0.7)
        else:
            color = colors_samples((i+offset)/(len(model_names)+offset-1))
        plt.plot(group_sizes, means[i], color=color, label=label[i], linewidth=4)
        plt.fill_between(group_sizes, q25s[i], q75s[i], color=color, alpha=0.1)
        # plt.errorbar(group_sizes, means[i], yerr=stds[i], label=model)
        plt.scatter(low_mean_x, low_means[i], color=color, label=None, s=50)
    plt.xscale('log', base=2)
    plt.xlim(left=1, right=low_mean_x+60)
    xticks = (1,2,4,8,16,32,64,128)
    plt.xticks(xticks+(low_mean_x,), xticks+(u'\u221e',))
    fig.axes[0].get_xticklabels()[-1].set_fontsize(22)
    offset = transforms.ScaledTranslation(0, 5/72, fig.dpi_scale_trans)
    last_label = fig.axes[0].xaxis.get_majorticklabels()[-1]
    last_label.set_transform(last_label.get_transform() + offset)
    plt.legend(loc='lower left')
    plt.yscale('log')
    yticks = [7, 5, 3, 2, 1, 0.6, 0.3]
    plt.yticks(yticks, yticks)
    plt.grid(True, which='both', axis='both', alpha=0.5)
    plt.savefig('{}.pdf'.format(plot_path), bbox_inches='tight')
    plt.savefig('{}.png'.format(plot_path), bbox_inches='tight')

def main(config_require, model_names, group_sizes, skip, dir, dev, plot_path):
    logging.info('Group sizes {}'.format(group_sizes))
    divs = read_configs(dir, config_require, dev)
    metrics = calc_metrics(divs, group_sizes, skip, len(model_names), main_model_i)
    plot(plot_path, group_sizes, model_names, main_model_i, metrics)

    for model, m in zip(model_names, metrics[0]):
        logging.info('Model: {}\n{}'.format(model, m))
    logging.info('Low means: {}'.format(metrics[-1]))

if __name__ == "__main__":
    plot_path = 'plots/groups'
    model_names = ['facebook/opt-{}'.format(x) for x in 
                    ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b']]
    main_model_i = 5
    response_len = 128
    prompt_len = 10
    dir = 'data'
    group_sizes = [i for i in range(1, response_len//2+1) if response_len%i<4] + [response_len] 
    config_require = {
        'model_names': model_names,
        'main_model_i': main_model_i,
        'n_tokens': response_len + prompt_len,
        'max_len': prompt_len,
    }
    dev = 'cuda:0'

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main(config_require, model_names, group_sizes, prompt_len, dir, dev, plot_path)