# Module 20: Recurrent Neural Networks (RNNs) & LSTMs

## 1. The Physical Intuition: Loops with Memory

Imagine reading a book one word at a time: *"The astronaut boarded the rocket ship."*

When you reach the word *"rocket"*, how do you know it refers to space travel? Because your brain holds a **persistent memory state** of the preceding words (*"astronaut"*, *"boarded"*). 

Standard Feedforward Neural Networks are completely stateless (`y = f(x)`). They process every input independently, wiping their memory completely between calls.

**Recurrent Neural Networks (RNNs)** are stateful processing loops (analogous to a `while` loop maintaining a persistent state accumulator variable across stream events):

```python
hidden_state = zeros()
for word in sentence:
    hidden_state = rnn_cell(word, hidden_state)
```

This allows RNNs to process variable-length sequential data (time-series stock ticks, log traces, sentence words). **LSTMs (Long Short-Term Memory)** add an explicit memory highway (Cell State) with gated read/write/erase controls to preserve long-range dependencies across thousands of sequence steps.

---

## 2. Core Concepts & Granular Step-by-Step Breakdown

### 1. Vanilla RNN & Vanishing Gradients
- Maintains hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.
- **Vanishing Gradient Bug**: Unrolling backpropagation through 100 time steps requires repeatedly multiplying by weight matrix $W_{hh}^{100}$. Gradients shrink exponentially to zero, forgetting early context!

---

### 2. LSTM (Long Short-Term Memory)
LSTMs fix vanishing gradients by introducing a **Cell State ($C_t$)** governed by three specialized sigmoidal gates:
1. **Forget Gate ($f_t$)**: Decides what percentage of old memory to erase (`0 = erase all`, `1 = keep all`).
2. **Input Gate ($i_t$)**: Decides what new incoming information to write into memory.
3. **Output Gate ($o_t$)**: Decides what subset of memory to expose as the visible hidden state output $h_t$.

---

## 3. Architecture & Visual Diagrams

### LSTM Cell Gated Architecture

```
Cell State C_{t-1} ───────────[ x ]──────────────────────[ + ]─────────────────► Cell State C_t
                              │                           ▲
                         Forget Gate f_t             Input Gate i_t
                              │                           │
Hidden State h_{t-1} ──┬──► [ Sigmoid ]            [ Sigmoid ] ──► Candidate C~_t
                       │                                  │
Input x_t ─────────────┴──► [ Sigmoid ] ──────────────────┴──► Output Gate o_t ──► Hidden State h_t
```

---

## 4. Practical Implementation: PyTorch LSTM Model

Let's write a PyTorch LSTM model configured for time-series forecasting or sequence classification:

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        last_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

# Test Run (Batch=16, Seq_Len=50, Features=5)
model = LSTMClassifier(input_dim=5, hidden_dim=64, num_layers=2, output_dim=1)
dummy_sequence = torch.randn(16, 50, 5)

predictions = model(dummy_sequence)
print(f"Input Sequence Batch Shape: {dummy_sequence.shape}")
print(f"Output Predictions Shape:   {predictions.shape}")
```

---

## 5. Demystifying the Mathematics: Explaining Every Formula

Let's dissect the LSTM update equations.

### Equation 1: Cell State Update Equation
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

- $f_t \odot C_{t-1}$: Erases unwanted old memories (multiplying by forget gate values $\in [0, 1]$).
- $i_t \odot \tilde{C}_t$: Adds new candidate memories (multiplying by input gate values $\in [0, 1]$).
- **Why addition ($\mathbf{+}$) is revolutionary**: Addition acts as a gradient highway; during backpropagation, gradients flow backwards through addition nodes without repeated matrix multiplication, eliminating vanishing gradients!

---

## 6. Real-World Production Gotchas & Failure Modes

### Bi-directional LSTMs & GRUs
- **BiLSTM**: Processes sequences forward and backward simultaneously, concatenating hidden states for full past/future context.
- **GRU (Gated Recurrent Unit)**: Combines cell state and hidden state, using only 2 gates (Reset & Update). Faster training with fewer parameters.

---

## 7. Feynman Exercises & Deep Thinking Challenges

1. **Thought Experiment**: Why do vanilla RNNs suffer from vanishing gradients while LSTMs do not?
   - *Answer/Explanation*: Vanilla RNNs compute hidden states using multiplicative recurrence $h_t = \tanh(W_{hh} h_{t-1} + \dots)$. Backpropagating over $T$ steps repeatedly multiplies by weight matrix $W_{hh}^T$, causing gradients to vanish. LSTMs update cell states using additive recurrence $C_t = f_t \odot C_{t-1} + \dots$, preserving gradient flow.

2. **Exercise**: Replace `nn.LSTM` with `nn.GRU` and compare total parameter counts.
