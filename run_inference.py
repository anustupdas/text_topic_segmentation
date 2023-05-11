import torch
from torch.utils.data import DataLoader
import json
import torch.nn.functional as F
from choiloader_emb import ChoiDataset, collate_fn, collate_fn_emb
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure
import os
import sys
from pathlib2 import Path
from custom_dutch_loader import CustomDutchDataset
from custom_dutch_loader_emb import CustomDutchEmbDataset

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()


def test(model, args, dataset):
    result_dict = {}
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                output_softmax = F.softmax(output, 1)
                out = output_softmax.data.cpu().numpy().argmax(axis=1)
                kutlu_put =(output_softmax.data.cpu().numpy()) [:, 1]
                out = kutlu_put > .5

                #result_dict[str(paths[0])] = output_seg.tolist()
                result_dict[str(paths[0])] = out.tolist()
                #print(f"for file {paths[0]} sentence labels are {output_seg}")
        return result_dict


def main(args):
    sys.path.append(str(Path(__file__).parent))

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None


    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        model = import_model(args.model)
    elif args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f,map_location=torch.device('cpu'))
    print(f"Here is model {model}")
    model.train()
    model = maybe_cuda(model)



    test_dataset = CustomDutchDataset(args.infer, word2vec=word2vec, inference=False,
                                    high_granularity=args.high_granularity)
    test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                         num_workers=args.num_workers)

    # test_dataset = CustomDutchEmbDataset(args.infer, word2vec=word2vec, inference=False,
    #                                   high_granularity=args.high_granularity)
    # test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn_emb, shuffle=False,
    #                      num_workers=args.num_workers)
    result = test(model, args, test_dl)
    with open(os.path.join(args.infer,'result_softmax.json'), 'w') as fp:
        json.dump(result, fp)
    print(f"Results saved in {args.infer} as results.json")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=10)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=1)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)

    main(parser.parse_args())
