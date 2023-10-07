python main.py --file_type="Smartphone-COQE" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"
python main.py --file_type="Smartphone-COQE" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/" --factor=0.3
python main.py --file_type="Smartphone-COQE" --model_mode="bert" --program_mode="test" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"

# run lstm crf
python main.py --file_type="Smartphone-COQE" --model_mode="norm" --program_mode="run" --stage_model="first" --epoch=25 --model_type="multitask" --embed_dropout=0.1 --premodel_path="/home/pretrain_model/"