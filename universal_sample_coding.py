import math
import itertools
import torch
from tqdm import tqdm


class Estimator:
    '''
    Implements discrete estimator from
    "Bernstein polynomials and learning theory" D. Braess, T. Sauer (5.18)
    '''
    def __init__(self, n_alphabet, batch=(), dev='cpu'):
        self.n_alphabet = n_alphabet
        self.batch = batch
        self.dev = dev
        size = batch + (n_alphabet,)
        self.count = torch.zeros(size, dtype=torch.int64, device=dev)
        self.total = torch.zeros(batch, dtype=torch.int64, device=dev)

    def update_count(self, x):
        assert x.shape == self.batch + (self.n_alphabet,)
        self.count += x
        self.total += x.sum(dim=-1)

    def prob(self):
        k_0 = self.count == 0
        k_1 = self.count == 1
        weight = self.count + 0.75 - 0.25 * k_0 + 0.25 * k_1
        return weight / weight.sum(dim=-1, keepdim=True)

class PriorEstimator:
    def __init__(self, prior, prior_scale, n_alphabet, batch=()):
        self.prior = prior
        self.prior_scale = prior_scale
        self.dev = prior.device
        self.est = Estimator(n_alphabet, batch=batch, dev=self.dev)

    def update_count(self, x):
        self.est.update_count(x) 

    def prob(self):
        w_est = self.est.prob() * self.est.total.unsqueeze(-1)
        posterior_w = w_est + self.prior * self.prior_scale 
        posterior = posterior_w / posterior_w.sum(dim=-1, keepdim=True)
        return posterior

class DirichletEstimator:
    def __init__(self, prior, prior_scale):
        self.dev = prior.device
        self.prior = prior
        self.prior_scale = prior_scale
        self.count = torch.zeros(prior.shape, dtype=torch.int64, device=prior.device)
        self.total = torch.zeros(prior.shape[:-1], dtype=torch.int64, device=prior.device)

    def update_count(self, x):
        assert x.shape == self.prior.shape
        self.count += x
        self.total += x.sum(dim=-1)

    def prob(self):
        params = self.prior * self.prior_scale + self.count
        # weights = (params + 0.75) / (self.total.unsqueeze(-1) + 1.25)
        # return weights / weights.sum(dim=-1, keepdim=True)
        return params / params.sum(dim=-1, keepdim=True)

class FixedEstimator:
    def __init__(self, probs):
        self.probs = probs
        self.dev = probs.device

    def update_count(self, x):
        pass

    def prob(self):
        return self.probs

class Prob:
    """
    Wrapper for torch.distributions.Categorical 
    adding a field specifying the number of samples
    """
    def __init__(self, probs, n_samples=1):
        self.probs = probs
        self.P = torch.distributions.Categorical(probs)
        self.batch = self.probs.shape[:-1]
        self.n_samples = n_samples

    def sample(self, shape):
        return self.P.sample(shape + (self.n_samples,))

    def log_prob(self, x):
        return self.P.log_prob(x).sum(dim=-1)

def KL(P, Q):
    return (P * ((P+1e-15) / Q).log2()).sum(dim=-1)

def ordered_random_coding_seq(P, Q, N, log_min_ratio=-torch.inf):
    t = 0
    w_log_min = float("inf")
    idx, sample = -1, -1
    e = torch.distributions.Exponential(1)
    for i in range(N):
        candidate = Q.sample(())
        v = N / (N - i)
        t += v * e.sample()
        log_t = torch.log(t)
        w_log = log_t + Q.log_prob(candidate) - P.log_prob(candidate)
        if w_log < w_log_min:
            w_log_min = w_log
            idx = i
            sample = candidate
        if w_log_min <= log_t + log_min_ratio:
            break
    return torch.tensor(idx), sample, i + 1

def ordered_random_coding_par(P, Q, N, log_min_ratio=None):
    dev = P.probs.device
    e = torch.distributions.Exponential(torch.tensor(1.0, device=dev))
    candidates = Q.sample((N,))
    exps = e.sample((N,))
    vs = N / (N - torch.arange(N, device=dev))
    ts = (vs * exps).cumsum(dim=0)
    w_logs = torch.log(ts) + (Q.log_prob(candidates) - P.log_prob(candidates)).view(-1)
    idx = w_logs.argmin()
    sample = candidates[idx].detach().clone()
    return idx, sample, N

def calc_batch(max_ram, sample_dimension):
    return max(max_ram // (int(sample_dimension) + 4), 1)

def ordered_random_coding(P, Q, N, log_min_ratio=-torch.inf, batch=-1):
    if batch == -1: 
        batch = N
    dev = P.probs.device
    e = torch.distributions.Exponential(torch.tensor(1.0, device=dev))
    w_log_min = torch.inf
    cum_t = 0
    for i in range(0, N, batch):
        size = min(batch, N-i)
        candidates = Q.sample((size,))
        exps = e.sample((size,))
        vs = N / (N - torch.arange(i, i+size, device=dev))
        ts = cum_t + (vs * exps).cumsum(dim=0)
        cum_t = ts[-1]
        w_logs = torch.log(ts) + (Q.log_prob(candidates) - P.log_prob(candidates)).view(-1)
        idx = w_logs.argmin()
        if w_logs[idx] < w_log_min:
            w_log_min = w_logs[idx]
            sample = candidates[idx].detach().clone()
        if w_log_min <= torch.log(cum_t) + log_min_ratio:
            break
    return idx, sample, i + size

def unif_code_len(idx, N):
    if type(N) == int:
        N = torch.tensor(N, device=idx.device)
    return torch.log2(N)

def zipf_log_prob(x, scale, N=None):
    if N is None:
        log_norm_const = torch.log(torch.special.zeta(scale, 1))
    else:
        log_norm_const = torch.log(torch.special.zeta(scale, 1) 
                                   - torch.special.zeta(scale, N+1))
    return -scale * torch.log(x) - log_norm_const

def zipf_code_len(idx, exp_KL, N=None):
    e_log2 = torch.log2(torch.tensor(torch.e, device=idx.device))
    scale = 1 + 1/(exp_KL + 1 + e_log2 / torch.e)
    log_prob = zipf_log_prob(idx + 1, scale, N=N) # idx + 1 because zipf is 1-indexed
    return -log_prob * e_log2

def all_code_length(idx, exp_KL, N):
    '''
    Calculates the code length for uniform, zipf, and bounded zipf distributions
    '''
    unif_len = unif_code_len(idx, N)
    zipf_len = zipf_code_len(idx, exp_KL)
    zipf_bounded_len = zipf_code_len(idx, exp_KL, N)
    return torch.stack([unif_len, zipf_len, zipf_bounded_len])

def usc_iterator(c, n):
    cum_samples = lambda j: math.ceil((1+c)**(j-1))
    round = 1
    sent_samples = 0
    total_samples = 1
    while True:
        yield round, total_samples - sent_samples
        sent_samples = total_samples
        round += 1
        total_samples = min(cum_samples(round), n)
        if sent_samples >= n:
            break

# sometimes usc_iterator returns a round with 0 samples, which we skip
def usc_iterator_skip_0(c, n):
    for round, samples in usc_iterator(c, n):
        if samples > 0:
            yield round, samples

def universal_sample_coding(P, estimator, N, sam_comm, coding_cost, c=1, 
                            MAX_EXP=14, use_KL=True, max_ram=-1, verbose=False):
    samples_list = []
    bits = []
    for round, n_samples in usc_iterator_skip_0(c, N):
        Q = Prob(estimator.prob())
        Q.n_samples = n_samples
        total_samples = torch.maximum(estimator.total, torch.tensor(1, device=P.probs.device))
        exp_nKL = n_samples * (estimator.n_alphabet - 1) / (2 * total_samples)
        nKL_PQ = n_samples * KL(P.probs, Q.probs)
        
        exp = nKL_PQ if use_KL else exp_nKL
        # exp = min(exp, MAX_EXP)
        max_iters = math.ceil(2**exp)
        
        if verbose:
            print('nKL {:5.2f}, exp_nKL {:4.1f}, max_iters: {:2.0e}, n_samples: {:3}'\
                  .format(nKL_PQ, exp_nKL, max_iters, n_samples), end='', flush=True)
        
        log_min_ratio = n_samples * torch.log((P.probs/Q.probs).min())
        batch_size = calc_batch(max_ram, estimator.n_alphabet * n_samples)
        idx, sample, iters = sam_comm(P, Q, max_iters, log_min_ratio=log_min_ratio, batch=batch_size)
        samples_list.append(sample)

        bit_len = coding_cost(idx, exp_nKL, N=max_iters)
        bits.append(bit_len)
        if verbose:
            print(', bits: {}, idx: {}, skip: {:.0e}'\
                  .format(tensor_str(bit_len, '{:3.1f}'), idx, max_iters-iters))
        sample_ohe = torch.bincount(sample, minlength=P.probs.shape[-1])
        estimator.update_count(sample_ohe)
    return samples_list, bits

def adaptive_sample_coding(P, estimator, N, KL_target):
    '''Universal Sample Coding with target KL oracle'''
    samples_list = []
    coding_costs = []
    samples_remain = N
    while True:
        Q = Prob(estimator.prob())
        KL_PQ = KL(P.probs, Q.probs)
        n_samples = min(math.ceil(KL_target / KL_PQ), samples_remain)
        Q.n_samples = n_samples
        nKL_PQ = n_samples * KL_PQ
        exp = nKL_PQ
        max_iters = math.ceil(2**exp)
        log_min_ratio = n_samples * torch.log((P.probs/Q.probs).min())

        idx, sample, iters = ordered_random_coding(P, Q, max_iters, log_min_ratio=log_min_ratio)
        samples_list.append(sample)
        coding_cost = all_code_length(idx, nKL_PQ, N=max_iters)
        coding_costs.append(coding_cost)  
        sample_ohe = torch.bincount(sample, minlength=P.probs.shape[0])
        estimator.update_count(sample_ohe)
        samples_remain -= n_samples
        if samples_remain <= 0:
            break
    return samples_list, coding_costs

def const_kl_blocks(kls, target_kl, max_block=-1):
    return block_group(kls, target_kl, max_block=max_block)

def aprox_group(x, val):
    x_blocks = torch.div(x.cumsum(dim=-1), val, rounding_mode='trunc')
    end_idxs = (x_blocks[1:] > x_blocks[:-1]).nonzero().squeeze(-1) + 1
    end_idxs = torch.cat([end_idxs, torch.tensor([x.shape[-1]], device=x.device)])
    return end_idxs

def block_group(x, val, max_block=-1):
    assert len(x.shape) == 1
    if max_block == -1:
        max_block = x.shape[0]
    x_larger = x > val
    x = x * ~x_larger + val * x_larger
    cum_x = torch.zeros(x.shape[0]+1, device=x.device)
    _ = torch.cumsum(x, dim=-1, out=cum_x[1:])

    end_idxs = []
    last_idx = torch.tensor(1, device=x.device)
    while True:
        search = val + cum_x[last_idx-1]
        idx = torch.searchsorted(cum_x, search, side='right')
        if idx - last_idx > max_block:
            idx = last_idx + max_block       
        end_idxs.append(idx.item())
        if idx == cum_x.shape[0]:
            break
        last_idx = idx
    return torch.tensor(end_idxs, device=x.device) - 1

def block_iterator(end_idxs):
    for start, end in zip(itertools.chain([0], end_idxs[:-1]), end_idxs):
        yield start, end

def batched_bincount(x, max_value):
    # https://discuss.pytorch.org/t/batched-bincount/72819/2
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def joint_universal_sample_coding(P, estimator, N, KL_target, coding_cost=None, 
                                  calc_KL=True, max_block=-1, max_ram=-1, 
                                  verbose=False):
    """Sample communication of N copies of many independent random varaibles."""
    samples_list = torch.zeros(P.probs.shape[:-1]+(N,), dtype=torch.int64, device=P.probs.device)
    bits = []
    for round in range(N):
        Q = Prob(estimator.prob())
        kls = KL(P.probs, Q.probs)
        block_samples = []
        blocks = list(const_kl_blocks(kls, KL_target, max_block=max_block))
        sample_idxs = torch.empty(len(blocks), dtype=torch.int64, device=P.probs.device)
        max_iters = torch.empty_like(sample_idxs)
        for i, (l_idx, r_idx) in tqdm(enumerate(block_iterator(blocks)), 
                                      total=len(blocks), leave=True, ncols=80,
                                      disable=not verbose):
            exp = kls[l_idx:r_idx].sum() if calc_KL else KL_target
            max_iter = math.ceil(2**exp)
            P_block = Prob(P.probs[l_idx:r_idx])
            Q_block = Prob(Q.probs[l_idx:r_idx])

            batch_size = calc_batch(max_ram, P_block.probs.shape[-1] * (r_idx-l_idx))
            idx, block_sample, _ = ordered_random_coding(P_block, Q_block, 
                                                         max_iter, batch=batch_size)
            block_samples.append(block_sample.view(-1))
            sample_idxs[i] = idx
            max_iters[i] = max_iter

        sample = torch.cat(block_samples)
        sample_ohe = batched_bincount(sample.unsqueeze(-1), P.probs.shape[-1])
        estimator.update_count(sample_ohe)
        samples_list[..., round] = sample

        if coding_cost is not None:
            bit_len = coding_cost(sample_idxs, KL_target, N=max_iters)
            bits.append(bit_len)
    
    bits = torch.cat(bits, dim=-1).sum(dim=-1) if coding_cost is not None else 0
    return samples_list, bits
 
def tensor_str(x, s):
    return '[' + ', '.join([s.format(i) for i in x]) + ']'

def simulate_universal_sample_coding(dev):
    dim = 4
    p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
    P = Prob(p)
    
    # for N in [1, 3, 8, 16, 24, 32]:
    for exp in range(1, 25):
        N = 2**exp
        est = Estimator(dim, dev=dev)
        print('\nNumber of samples: {}'.format(N))
        samples, bits = universal_sample_coding(P, est, N, ordered_random_coding, 
                                                all_code_length, c=3, 
                                                verbose=True,max_ram=2**30)
        samples = torch.cat(samples)
        counts = torch.bincount(samples, minlength=dim)
        freq = counts / counts.sum()
        empirical_KL = KL(freq, p)
        print('freq: {}'.format(freq))
        print('empirical_KL: {}'.format(empirical_KL))
        coding_cost = torch.stack(bits).sum(dim=0)
        print(coding_cost)
        coding_cost_lowerbound = (dim-1)/2 * math.log2(N)
        print('coding_cost: {}, per sample: {}, lower bound: {:.0f}'\
              .format(tensor_str(coding_cost, '{:.5f}'), 
                        tensor_str(coding_cost/N, '{:.5f}'),
                        coding_cost_lowerbound))

if __name__ == "__main__":
    dev = torch.device("cuda:1")
    simulate_universal_sample_coding(dev) 