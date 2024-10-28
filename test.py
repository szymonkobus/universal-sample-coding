import unittest
import torch
from universal_sample_coding import Estimator, KL, batched_bincount,  \
    block_iterator, block_group, ordered_random_coding_seq, \
    ordered_random_coding_par, ordered_random_coding, Prob, zipf_log_prob, \
    zipf_code_len, usc_iterator, usc_iterator_skip_0, universal_sample_coding, \
    aprox_group, joint_universal_sample_coding


def fix_seed(seed=2049):
    def _fix_seed(f):
        def wrapper(*args, **kwds):
            curr_seed = torch.seed()
            torch.manual_seed(seed)
            res = f(*args, **kwds)
            torch.manual_seed(curr_seed)
            return res
        return wrapper
    return _fix_seed

class USCtest(unittest.TestCase):
    def arr_to_tensor(self, arr, dev):
        return torch.tensor(arr, device=dev)

    def test_estimator(self):
        dev = 'cuda'
        freq = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [2, 1, 0, 0],
            [0, 0, 0, 3],
            [2, 2, 0, 0],
        ]
        est = Estimator(4, batch=(len(freq),), dev=dev)
        est.update_count(self.arr_to_tensor(freq, dev=dev))

        exp_weigth = [
            [1, 1, 1, 1],
            [4, 1, 1, 1],
            [1, 4, 4, 1],
            [11, 8, 2, 2],
            [2, 2, 2, 15],
            [11, 11, 2, 2],
        ]
        exp_weigth = self.arr_to_tensor(exp_weigth, dev=dev)
        exp_prob = exp_weigth / exp_weigth.sum(dim=-1, keepdim=True)

        prob = est.prob()
        self.assertTrue(torch.equal(prob, exp_prob))

    @fix_seed(2048)
    def test_estimator_permutation(self):
        dev = 'cuda'
        dim = 20
        batch = 100
        est = Estimator(dim, batch=(batch,), dev=dev)
        freq = torch.randint(0, 100, (dim,), device=dev)
        rand_val = torch.rand(batch, dim, device=dev)
        indices = torch.argsort(rand_val, dim=-1)
        freqs = freq[indices]
        est.update_count(freqs)
        prob = est.prob()
        sorted_prob, _ = torch.sort(prob, dim=-1)
        self.assertTrue(torch.all(torch.abs(sorted_prob[1:]-sorted_prob[:-1])<1e-7))

    @fix_seed(2049)
    def test_ordered_random_coding_seq(self):
        for dev in ['cpu', 'cuda']:
            p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
            q = torch.tensor([0.25, 0.25, 0.25, 0.25], device=dev)
            P = torch.distributions.Categorical(p)
            Q = torch.distributions.Categorical(q)
            ratio_min = torch.log((p/q).min())
            counts = torch.zeros_like(p)
            for _ in range(100):
                idx, sample, iters = ordered_random_coding_seq(P, Q, 20, ratio_min)
                counts[sample] += 1
            freq = counts/counts.sum()
            empirical_KL = KL(freq, p)
            self.assertTrue(empirical_KL < 0.03)

    @fix_seed(2050)
    def test_ordered_random_coding_par(self):
        for dev in ['cpu', 'cuda']:
            p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
            q = torch.tensor([0.25, 0.25, 0.25, 0.25], device=dev)
            P = torch.distributions.Categorical(p)
            Q = torch.distributions.Categorical(q)
            counts = torch.zeros_like(p)
            for _ in range(100):
                idx, sample, iters = ordered_random_coding_par(P, Q, 20)
                counts[sample] += 1
            freq = counts/counts.sum()
            empirical_KL = KL(freq, p)
            self.assertTrue(empirical_KL < 0.03)

    @fix_seed(2051)
    def test_ordered_random_coding(self):
        for dev in ['cuda']:
            p = torch.tensor([0.04, 0.03, 0.03, 0.9], device=dev)
            q = torch.tensor([0.25, 0.25, 0.30, 0.20], device=dev)
            P = torch.distributions.Categorical(p)
            Q = torch.distributions.Categorical(q)
            counts = torch.zeros_like(p)
            log_min_ratio = torch.log((p/q).min())
            for _ in range(100):
                _, sample, _ = ordered_random_coding(P, Q, 1000, 
                                                     log_min_ratio=log_min_ratio)
                counts[sample] += 1
            freq = counts/counts.sum()
            empirical_KL = KL(freq, p)
            self.assertTrue(empirical_KL < 0.03)

    def test_Prob(self):
        for dev in ['cpu', 'cuda']:
            p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
            P = Prob(p)
            P.n_samples = 100
            x = P.sample((200,))
            self.assertTrue(x.shape == (200, 100))

    @fix_seed(2052)
    def test_Prob_and_ordered_random_coding_par(self):
        n_samples = 100
        N = 2**16
        for dev in ['cpu', 'cuda']:
            p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
            q = torch.tensor([0.25, 0.25, 0.25, 0.25], device=dev)
            P = Prob(p, n_samples=n_samples)
            Q = Prob(q, n_samples=n_samples)
            idx, sample, iters = ordered_random_coding_par(P, Q, N)
            counts = torch.bincount(sample, minlength=4)
            freq = counts / counts.sum()
            empirical_KL = KL(freq, p)
            self.assertTrue(empirical_KL < 0.03)

    def test_zipf_log_prob(self):
        x = torch.tensor(7)
        scale = torch.tensor(2)
        norm_const = torch.arange(1, 1000).float().pow(-scale).sum()
        exp_prob = x.float()**(-scale) / norm_const
        prob = zipf_log_prob(x, scale).exp()
        self.assertTrue(torch.allclose(exp_prob, prob, atol=1e-5))

    def test_zipf_log_prob_bound(self):
        x = torch.arange(1, 5)
        N = torch.tensor([4] * x.shape[0])
        scale = torch.tensor(2)
        probs = zipf_log_prob(x, scale, N).exp()
        self.assertTrue(torch.allclose(probs.sum(), torch.tensor(1.0)))

    def test_zipf_coding_cost(self):
        exp_KL = torch.tensor(10.0)
        idx = torch.arange(0, 1_000, device='cuda')
        cost = zipf_code_len(idx, exp_KL)
        # TODO: how to test the cost is right.

    def test_usc_iterator_1(self):
        rounds, n_sampless = zip(*usc_iterator(1, 128))
        exp_rounds = tuple(range(1, 9))
        exp_n_samples = (1, 1, 2, 4, 8, 16, 32, 64)
        self.assertTrue(rounds == exp_rounds)
        self.assertTrue(n_sampless == exp_n_samples)

    def test_usc_iterator_2(self):
        c, N = 0.1, 2048
        _, n_sampless_1 = zip(*usc_iterator(c, N))
        _, n_sampless_2 = zip(*usc_iterator_skip_0(c, N))
        self.assertTrue(sum(n_sampless_1) == sum(n_sampless_2))

    @unittest.skip
    @fix_seed(2056)
    def test_universal_sample_coding(self):
        dev = 'cpu'
        dim = 4
        p = torch.tensor([0.1, 0.2, 0.3, 0.4], device=dev)
        P = Prob(p)
        for exponent in range(1, 10):
            estimator = Estimator(dim, dev=dev)
            N = 2**exponent
            samples_list, coding_costs = universal_sample_coding(
                P, estimator, N, ordered_random_coding_seq, zipf_code_len
            )

            samples_list = torch.cat(samples_list)
            counts = torch.bincount(samples_list, minlength=dim)
            freq = counts / counts.sum()
            empirical_KL = KL(freq, p)
            exp_kl = (dim-1)/(2*N)
            if N > 4:
                self.assertTrue(empirical_KL < 3 * exp_kl)

    def test_aprox_group(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([1, 1, 1, 2, 2, 2, 0.5, 0.2, 3, 4], device=dev)
            idxs = aprox_group(x, 3.51)
            exp_idxs = torch.tensor([3, 5, 8, 9, 10], device=dev)
            self.assertTrue(torch.all(idxs == exp_idxs))

    def test_block_group(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([1, 1, 1, 2, 2, 2, 0.5, 0.2, 3, 1], device=dev)
            idxs = block_group(x, 3.51)
            exp_idxs = torch.tensor([3, 4, 5, 8, 9, 10], device=dev)
            self.assertTrue(torch.all(idxs == exp_idxs))

    def test_block_group_big(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([4, 1, 4, 1, 1, 4], device=dev)
            idxs = block_group(x, 3.51)
            exp_idxs = torch.tensor([1, 2, 3, 5, 6], device=dev)
            self.assertTrue(torch.all(idxs == exp_idxs))

    def test_block_group_max_block(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([1, 1, 1, 2, 2, 2, 0.5, 0.2, 3, 4], device=dev)
            idxs = block_group(x, 3.51, max_block=2)
            exp_idxs = torch.tensor([2, 4, 5, 7, 9, 10], device=dev)
            self.assertTrue(torch.all(idxs == exp_idxs))

    def test_block_iterator(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([1, 1, 1, 2, 2, 2, 0.5, 0.2, 3, 4], device=dev)
            idxs = torch.tensor([3, 4, 5, 8, 9, 10], device=dev)
            exp_blocks = [[1, 1, 1], [2], [2], [2, 0.5, 0.2], [3], [4]]
            for (l, r), exp_block in zip(block_iterator(idxs), exp_blocks):
                self.assertTrue(torch.all(x[l:r] == torch.tensor(exp_block, device=dev)))

    def test_batched_bincount(self):
        for dev in ['cpu', 'cuda']:
            x = torch.tensor([[1, 1, 2], [0, 0, 0]], device=dev)
            bins = batched_bincount(x, 3)
            exp_bin = torch.tensor([[0, 2, 1], [3, 0, 0]], device=dev)
            self.assertTrue(torch.all(bins == exp_bin))

    @fix_seed(2058)
    def test_joint_universal_sample_coding(self):
        dev = 'cpu'
        P = Prob(torch.tensor([[0.3, 0.3, 0.4], [0.9, 0.05, 0.05]], device=dev))
        est = Estimator(P.probs.shape[-1], batch=P.probs.shape[:-1], dev=dev)
        KL_target = 4
        N = 10
        _, _ = joint_universal_sample_coding(P, est, N, KL_target, calc_KL=False)
        kl = KL(P.probs, est.prob())
        self.assertTrue(torch.all(kl < torch.tensor(3/N)))


# python -m unittest discover -s . -p "*test*" -v