import torch
import torch.nn
import model
import sentencepiece as spm
import config as cfg
import os

def load_vocab(file):
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab

if __name__ == '__main__' :
    model_dir = "C:/Users/ys499/Desktop/DL_implement/Transformer/save_best.pth"
    config_dir = "C:/Users/ys499/Desktop/DL_implement/Transformer/config.json"
    vocab = load_vocab("C:/Users/ys499/transformer_data/web-crawler/kowiki/kowiki.model")
    config = cfg.Config(config_dir)
    config.n_enc_vocab, config.n_dec_vocab = len(vocab), len(vocab)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformer = model.MovieClassification(config)
    if os.path.isfile(model_dir):
        best_epoch, best_loss, best_score = transformer.load(model_dir)
    print(f"load state dict from: {model_dir}")
    transformer.eval()
    inputsentence = ['0']
    while 1:
        inputsentence[0] = input()
        piece = vocab.encode_as_pieces(inputsentence[0])
        id = vocab.encode_as_ids(inputsentence[0])
        enc_input = torch.tensor([id]) 
        dec_input = torch.tensor([vocab.encode_as_ids("[BOS]")])
        output = transformer(enc_input, dec_input)
        # print(output.size())
        logit = output[0]
        _, index = logit.max(1)
        if index == 0 :
            print("Negative")
        elif index == 1 :
            print("Positive")
        else :
            print("Error")
