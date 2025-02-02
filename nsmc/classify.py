import sys
import argparse

import torch
import torch.nn as nn

import torchtext
from models.rnn import RNNClassifier
from models.cnn import CNNClassifier
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data

def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)

    p.add_argument('--drop_rnn', action='store_true') # store_true : 인자를 적으면(값을 주지 않는다) 해당 인자에 true나 false가 저장된다.
    # store_true의 경우 default 값은 false이며, 인자를 적어 주면 true가 저장된다.
    # store_false의 경우 반대이다.
    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config

def read_text(max_length=256):
    '''
    Read text from standard input for inference.
    '''

    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines.append(line.strip().split(' ')[:max_length])
    
    return lines

def define_field():

    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )
    # print(f'saved_data: {saved_data}')

    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=config.max_length)

    with torch.no_grad():
        ensemble = []

        if rnn_best is not None and not config.drop_cnn:
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                hidden_size=train_config.hidden_size,
                n_classes=n_classes,
                n_layers=train_config.n_layers,
                dropout_p=train_config.dropout,
            )
            model.load_state_dict(rnn_best)
            ensemble.append(model)
        if cnn_best is not None and not config.drop_cnn:
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=train_config.use_batch_norm,
                dropout_p=train_config.dropout,
                window_sizes=train_config.window_sizes,
                n_filters=train_config.n_filters,
            )
            model.load_state_dict(cnn_best)
            ensemble.append(model)

        y_hats = []

        # Get prediction with iteration on ensemble.
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)

            model.eval()

            y_hat = []
            for idx in range(0, len(lines), config.batch_size):
                x = text_field.numericalize(
                    text_field.pad(lines[idx:idx + config.batch_size]),
                    device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
                )
                y_hat.append(model(x).cpu())

            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats.append(y_hat)

            model.cpu()
        y_hats = torch.stack(y_hats).exp()
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.mean(dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.topk(config.top_k, dim=-1)

        for i in range(len(lines)):

            sys.stdout.write(
                f"{' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)])}\t{' '.join(lines[i])}\n"
            )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
    
            

