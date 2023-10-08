# import pickle
# from collections import Counter
#
# import torch
# import json
# import numpy as np
#
# import random
# import os
# import argparse
# import Config
#
# from data_utils import shared_utils, kesserl14_utils, coae13_utils, data_loader_utils
# from model_utils import train_test_utils
# from eval_utils.base_eval import BaseEvaluation, ElementEvaluation, PairEvaluation
# from eval_utils import create_eval
# from data_utils import current_program_code as cpc
# from tqdm import tqdm
# from transformers import AutoModel
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.nn.Module.dump_patches = True
#
#
# def TerminalParser():
#     # define parse parameters
#     parser = argparse.ArgumentParser()
#     parser.description = 'choose train data and test data file path'
#
#     parser.add_argument('--seed', help='random seed', type=int, default=2021)
#     parser.add_argument('--batch', help='input data batch size', type=int, default=16)
#     parser.add_argument('--epoch', help='the number of run times', type=int, default=25)
#     parser.add_argument('--fold', help='the fold of data', type=int, default=5)
#
#     # lstm parameters setting
#     parser.add_argument('--input_size', help='the size of encoder embedding', type=int, default=300)
#     parser.add_argument('--hidden_size', help='the size of hidden embedding', type=int, default=512)
#     parser.add_argument('--num_layers', help='the number of layer', type=int, default=2)
#
#     # program mode choose.
#     parser.add_argument('--model_mode', help='bert or norm', default='bert')
#     parser.add_argument('--server_type', help='1080ti or rtx', default='1080ti')
#     parser.add_argument('--program_mode', help='debug or run or test', default='run')
#     parser.add_argument('--stage_model', help='first or second', default='first')
#     parser.add_argument('--model_type', help='bert_crf, bert_crf_mtl', default='crf')
#     parser.add_argument('--position_sys', help='BIES or BI or SPAN', default='BMES')
#
#     parser.add_argument('--device', help='run program in device type',
#                         default='cpu' if torch.cuda.is_available() else 'cpu')
#
#     parser.add_argument('--file_type', help='the type of data set', default='car')
#     parser.add_argument('--premodel_path', help='the type of data set', default=None)
#
#     # model parameters.
#     parser.add_argument('--embed_dropout', help='prob of embedding dropout', type=float, default=0.1)
#     parser.add_argument('--factor', help='the type of data set', type=float, default=0.4)
#
#     # optimizer parameters.
#     parser.add_argument('--bert_lr', help='the type of data set', type=float, default=2e-5)
#     parser.add_argument('--linear_lr', help='the type of data set', type=float, default=2e-5)
#     parser.add_argument('--crf_lr', help='the type of data set', type=float, default=0.01)
#
#     args = parser.parse_args()
#
#     return args
#
#
# def get_necessary_parameters(args):
#     """
#     :param args:
#     :return:
#     """
#     param_dict = {"file_type": args.file_type,
#                   "model_mode": args.model_mode,
#                   "stage_model": args.stage_model,
#                   "model_type": args.model_type,
#                   "epoch": args.epoch,
#                   "batch_size": args.batch,
#                   "program_mode": args.program_mode}
#
#     return param_dict
#
#
# def main():
#     # get program configure
#     args = TerminalParser()
#
#     # set random seed
#     set_seed(args.seed)
#
#     config = Config.BaseConfig(args)
#     config_parameters = get_necessary_parameters(args)
#
#     if args.stage_model == "first":
#         model_parameters = {"embed_dropout": args.embed_dropout}
#     else:
#         model_parameters = {"embed_dropout": args.embed_dropout, "factor": args.factor}
#
#     optimizer_parameters = None
#
#     model_name = shared_utils.parameters_to_model_name(
#         {"config": config_parameters, "model": model_parameters}
#     )
#
#     print(model_name)
#     print(config)
#     if config.data_type == "eng" or config.data_type == "vie":
#         data_gene = kesserl14_utils.DataGenerator(config)
#     else:
#         data_gene = coae13_utils.DataGenerator(config)
#
#     data_gene.generate_data()
#
#     print("create data loader")
#     train_loader = data_loader_utils.create_first_data_loader(
#         data_gene.train_data_dict, config.batch_size
#     )
#
#     dev_loader = data_loader_utils.create_first_data_loader(
#         data_gene.dev_data_dict, config.batch_size
#     )
#
#     test_loader = data_loader_utils.create_first_data_loader(
#         data_gene.test_data_dict, config.batch_size
#     )
#
#     model = AutoModel.from_pretrained("Fsoft-AIC/videberta-base").to(config.device)
#
#     for index, data in tqdm(enumerate(train_loader)):
#         input_ids, attn_mask, comparative_label, multi_label, result_label = data
#
#         input_ids = torch.tensor(input_ids).long().to(config.device)
#         attn_mask = torch.tensor(attn_mask).long().to(config.device)
#
#         token_embedding = model(input_ids, attention_mask=attn_mask)[0]
#         with open('train_representations.txt', 'ab') as file:
#             representations = token_embedding.detach().numpy()
#             pickle.dump(representations, file)
#
#     for index, data in tqdm(enumerate(dev_loader)):
#         input_ids, attn_mask, comparative_label, multi_label, result_label = data
#
#         input_ids = torch.tensor(input_ids).long().to(config.device)
#         attn_mask = torch.tensor(attn_mask).long().to(config.device)
#
#         token_embedding = model(input_ids, attention_mask=attn_mask)[0]
#         with open('dev_representations.txt', 'ab') as file:
#             representations = token_embedding.detach().numpy()
#             pickle.dump(representations, file)
#
#     for index, data in tqdm(enumerate(test_loader)):
#         input_ids, attn_mask, comparative_label, multi_label, result_label = data
#
#         input_ids = torch.tensor(input_ids).long().to(config.device)
#         attn_mask = torch.tensor(attn_mask).long().to(config.device)
#
#         token_embedding = model(input_ids, attention_mask=attn_mask)[0]
#         with open('test_representations.txt', 'ab') as file:
#             representations = token_embedding.detach().numpy()
#             pickle.dump(representations, file)
#
# if __name__ == "__main__":
#     main()
