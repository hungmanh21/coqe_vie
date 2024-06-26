from data_utils import shared_utils
from transformers import BertTokenizer, AutoTokenizer


class BaseConfig(object):
    def __init__(self, args):
        # common parameters
        self.epochs = args.epoch
        self.batch_size = args.batch
        self.device = args.device
        self.fold = args.fold

        # lstm model parameters setting
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers

        # common model mode setting
        self.model_mode = args.model_mode
        self.model_type = args.model_type
        self.file_type = args.file_type
        self.stage_model = args.stage_model
        self.program_mode = args.program_mode
        self.position_sys = args.position_sys

        self.premodel_path = args.premodel_path

        self.data_type = "eng" if args.file_type == "Camera-COQE" else "vie"

        self.path = PathConfig(
            self.device, self.file_type, self.program_mode, self.premodel_path
        )
        self.val = GlobalConfig(self.position_sys)
        print('self.path.bert_model_path', self.path.bert_model_path)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(self.path.bert_model_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.path.bert_model_path)

class PathConfig(object):
    def __init__(self, device, file_type, program_mode, premodel_path):
        # store split train and test data file path
        dir_name = file_type
        dir_name = dir_name if program_mode in {"run", "test", "dev"} else "test_" + dir_name

        self.standard_path = {
            "train": "../data/{}/train.txt".format(dir_name),
            "dev": "../data/{}/dev.txt".format(dir_name),
            "test": "../data/{}/test.txt".format(dir_name)
        }

        # nlp tool file path
        if device == "cpu":
            self.stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
            self.bert_model_path = "vinai/phobert-base-v2" if file_type == "Smartphone-COQE" else r"bert-base-uncased"
        else:
            self.stanford_path = r"D:/stanford-corenlp-full-2018-10-05"
            self.bert_model_path = "vinai/phobert-base-v2" if file_type == "Smartphone-COQE" else r"bert-base-uncased"

        self.pre_process_data = {
            "train": "../data/pre_process/{}_train_data.txt".format(file_type),
            "dev": "../data/pre_process/{}_dev_data.txt".format(file_type),
            "test": "../data/pre_process/{}_test_data.txt".format(file_type)
        }


class GlobalConfig(object):
    def __init__(self, position_sys):
        self.elem_col = ["entity_1", "entity_2", "aspect", "result"]
        self.polarity_col = ["Negative", "Equal", "Positive", "None"]
        self.polarity_dict = {k: index - 1 for index, k in enumerate(self.polarity_col)}

        if position_sys == "SPAN":
            self.position_sys = []
        else:
            self.position_sys = list(position_sys)

        self.special_id_map, self.norm_id_map = {"O": 0}, {"O": 0}

        self.norm_id_map = shared_utils.create_tag_mapping_ids([], self.position_sys, other_flag=True)
        self.special_id_map = shared_utils.create_tag_mapping_ids(self.polarity_col, self.position_sys, other_flag=True)

        self.invert_special_id_map = {v: k for k, v in self.special_id_map.items()}
        self.invert_norm_id_map = {v: k for k, v in self.norm_id_map.items()}
