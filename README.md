# GPT-2 Token Embeddings Visualization

This project visualizes the token embeddings of GPT-2 in 3D space, reduced from 768 dimensions.

## How to Run

### Option 1: Run the Notebook

You can run the notebook directly in an environment like Google Colab.

### Option 2: Local Setup

1. Install the required dependencies:
    ```bash
    pip install jupyter transformers tensorflow tensorboard tqdm torch
    ```

2. Start TensorBoard to visualize the embeddings:
    ```bash
    tensorboard --logdir=./logs/vocab/
    ```

3. Open your browser and go to [http://localhost:6006/](http://localhost:6006/) to see the visualization. You can search for a word and isolate it to, for example, the 20 nearest neighbors to get a better understanding of the relationships between words.

Now, you're ready to explore the token embeddings of GPT-2 in 3D space!

Credits to: [Visualizing GPT2 Word Embeddings on TensorBoard by Taaniya Arora](https://medium.com/@TaaniyaArora/visualizing-gpt2-word-embeddings-on-tensorboard-ea5c8fef9efa)
