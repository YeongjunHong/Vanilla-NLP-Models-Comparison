from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_and_preprocess_data(batch_size=16):
    # Load dataset
    dataset = load_dataset("imdb")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Tokenize datasets
    tokenized_datasets = {}
    for split in ['train', 'test']:
        tokenized_datasets[split] = dataset[split].map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
        # Rename 'label' to 'labels' for consistency with PyTorch conventions
        tokenized_datasets[split] = tokenized_datasets[split].rename_column("label", "labels")

    # Prepare DataLoaders
    train_dataset = tokenized_datasets['train'].with_format("torch")
    test_dataset = tokenized_datasets['test'].with_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(tokenizer.vocab)
