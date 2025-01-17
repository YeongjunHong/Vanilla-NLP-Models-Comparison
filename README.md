# Vanilla NLP Model Comparison

This project aims to compares among NLP models; from the traditional one(RNN) to new gen babies(Transformer, Bert, and GPT) 
on the IMDb movie reviews dataset.
I tried my best to import less libraries and make the models lightweight , so that they are runnable not only on GPU devices but,  CPU only environments. 


## Further update will be regarding as below
    1. Model Performance Comparison Across Different Embedding Types
    2. Comparison of Model Performance with and without Pre-Trained Embeddings 


## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YeongjunHong/Vanilla-NLP-Model-Comparison.git
    cd Vanilla-NLP-Model-Comparison
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run main.py:**
    ```bash
    python main.py
    ```

## Project Structure

- `data_loading.py`: Script for loading and preprocessing the IMDb dataset.
- `training.py`: Script for training and evaluating the models.
- `models.py`: Define different vanilla NLP models for the comparison. RNN, Transformer, Bert, GPT are used here.
- `embeddings.py`: will be added later on
- `main.py`: Script for execution.
- `requirements.txt`: List of required libraries.
- `README.md`: Project overview and setup instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
