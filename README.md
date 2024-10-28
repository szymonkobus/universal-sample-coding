# Universal Sample Coding

Code for "Universal Sample Coding" by Szymon Kobus, Tze-Yang Tung, and Deniz Gunduz, NeurIPS 2024.

### Files:
- `universal_sample_coding.py` - Implementation of Universal Sample Coding.
- `test.py` - Tests for Universal Sample Coding implementation.
- `sample.py` - Generates and stores data for the LLM experiment.
- `process_sample.py` - Processes and plots LLM experiment data.
- `find_base_USC.py` - Finds $\inf_c V_k(c)$ for dimension $k$.

### Replicating Results:

To replicate the communication cost per token for different OPT models, run:

```bash
python sample.py && \
python process_sample.py
```

To calculate and plot the minimal universal coding factor $V_k(c)$, run:

```bash
python find_base_USC.py
```

### Software and Libraries Used:
- `Python` 3.11
- `PyTorch` 2.1
- `NumPy`
- `SciPy`
- Huggingface `datasets`
- Huggingface `transformers`
- `Matplotlib`

### Dataset Used:
- OpenBookQA
