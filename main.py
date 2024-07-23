from data_loading import load_and_preprocess_data
from models import RNNModel, TransformerModel, BERTModel, GPTModel
from training import train_model
import os
os.environ['OMP_NUM_THREADS'] = '1'

def main():
    # Load data
    train_loader, test_loader, vocab_size = load_and_preprocess_data(batch_size=8)

    # Initialize models
    rnn_model = RNNModel(vocab_size, embedding_dim=100, hidden_dim=256)
    transformer_model = TransformerModel(vocab_size, d_model=256, nhead=8, num_layers=3)
    bert_model = BERTModel()
    gpt_model = GPTModel()

    models = [
        ("RNN", rnn_model),
        ("Transformer", transformer_model),
        ("BERT", bert_model),
        ("GPT", gpt_model)
    ]

    # Train and evaluate each model
    for model_name, model in models:
        print(f"Training {model_name} model...")
        train_model(model_name, model, train_loader, test_loader)
        print("\n")

if __name__ == "__main__":
    main()