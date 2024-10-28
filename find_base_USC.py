import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from scipy.optimize import minimize
from tqdm import tqdm


def v(c, dim):
    # Calculate factor V_k(c)
    redundancy = c * (dim-1) / 2
    overhead = log((dim-1) / (2*log(2)) + 1) + 5*log(2)
    return (redundancy + overhead) / log(1+c) + 1

def minimize_v(dim):
    # Calcualte the optimal value of c for a given dimension, value of V(c) 
    # the lowerbound (dim-1)/2 and V(c) normalized by the lowerbound
    norms = [(dim-1)/2 for dim in dims]
    cs, vs, vs_norm = [], [], []
    c_guess = 1.0
    for dim, norm in tqdm(zip(dims, norms), total=len(dims)):
        res = minimize(lambda c: v(c, dim), c_guess, bounds=[(0, None)])#, method='SLSQP')
        c_guess = res.x[0]
        cs.append(res.x[0])
        vs.append(res.fun)
        vs_norm.append(res.fun / norm)
        logging.info('dim: {:5}, c: {:.3f}, v: {:7.3f}, norm: {:.3f}'\
              .format(dim, cs[-1], vs[-1], vs_norm[-1]))
    return cs, vs, vs_norm, norms

def plot(plot_path, dims, vs, vs_norm, norm):
    plt.rcParams.update({'font.size': 14})
    color_v = plt.colormaps["Blues"](0.8)
    color_k = plt.colormaps["Oranges"](0.65)

    # Plot V(c) and the lowerbound
    plt.figure()
    plt.xlabel('$k$ (dimension)')
    plt.plot(dims, vs, color=color_v, label='$\\inf_{c} V_k(c)$', linewidth=4)
    plt.plot(dims, norm, color=color_k, label='$\\frac{k-1}{2}$', linewidth=4)
    plt.xscale('log', base=2)
    plt.xlim(left=2, right=dims[-1])
    xticks = [2**i for i in range(1, 12)]
    xlabels = xticks[:7] + ['$2^{' + str(i) + '}$'for i in range(1, 12)][7:]
    plt.xticks(xticks, xlabels)
    plt.yscale('log', base=2)
    plt.ylim(bottom=0.5)
    yticks = [0.5] + [2**i for i in range(0, 11)]
    plt.yticks(yticks, yticks)
    plt.grid(True, which='major', axis='both', alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig(plot_path + '.png', bbox_inches='tight')
    plt.savefig(plot_path + '.pdf', bbox_inches='tight')
    plt.close()

    # Plot normalized V(c)
    plt.figure()
    plt.xlabel('$k$ (dimension)')
    plt.plot(dims, vs_norm, color=color_v, label='$\\frac{2}{k-1}\\inf_{c} V_k(c)$', linewidth=4)
    plt.plot(dims, [1 for _ in vs_norm], color=color_k, label='$1$', linewidth=4)
    plt.xscale('log', base=2)
    plt.xlim(left=2, right=dims[-1])
    xticks = [2**i for i in range(1, 12)]
    xlabels = xticks[:7] + ['$2^{' + str(i) + '}$'for i in range(1, 12)][7:]
    plt.xticks(xticks, xlabels)
    plt.yscale('log', base=2)
    yticks = list(range(1,11))
    ylabels = [1, 2, 3, 4, '', 6, '', 8, '', 10]
    plt.yticks(yticks, ylabels)
    plt.grid(True, which='major', axis='both', alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig(plot_path + '_norm.png', bbox_inches='tight')
    plt.savefig(plot_path + '_norm.pdf', bbox_inches='tight')
    plt.close()

def main(dims, plot_path):
    cs, vs, vs_norm, norms = minimize_v(dims)
    for c, dim in zip(cs, dims):
        logging.info('dim: {:5}, c: {:.3f}'.format(dim, c))
    # plot(plot_path, dims, vs, vs_norm, norms)

if __name__ == '__main__':
    plot_path = 'plots/base_USC'
    l, r, N = 2, 2**11, 1000
    points = [2**i for i in np.linspace(np.log2(l), np.log2(r), N)]
    # dims = np.array(sorted(points + list(range(l, r))))
    dims = np.array(list(range(2, 50)))
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main(dims, plot_path)