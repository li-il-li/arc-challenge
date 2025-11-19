# Research Log

### 19.11.25, Wed

### Task

- Training Loop

#### Learnings

- One *epoch* is one entire passthrough of the training dataset
- Learning rate decay is often bound to epochs
- `def __call__` makes an boject callable
- BERT outputs predictions for all word (input length) but only MASK are considered

### 18.11.25, Tue

### Task

- Dataset via Huggingface + Train, Test, Validation Split
- Model and Tokkenizer via Huggingface Pytorch compatible setup

#### Learnings

- Tokkenizer runs on CPU but you have to load output .to GPU