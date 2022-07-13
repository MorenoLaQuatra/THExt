import os
import gc
import random
import logging
from typing import List

import numpy as np 
import pandas as pd 
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


class InternalDataset(Dataset):
    def __init__(self, 
                text_list: List[str], 
                context_list: List[str], 
                labels: torch.tensor=None, 
                max_length: int=384,
                tokenizer=None):
        super(InternalDataset, self).__init__()
        self.text_list = text_list
        self.context_list = context_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels


    def __getitem__(self, idx):
        
        out = self.tokenizer.encode_plus(self.text_list[idx],
                                        self.context_list[idx],
                                        padding="max_length",
                                        max_length=self.max_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='pt',
                                        return_token_type_ids=True)
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        token_type_ids = out["token_type_ids"]
        if self.labels is None: # test
            return input_ids[0], attention_mask[0], token_type_ids[0]
        else:
            return input_ids[0], attention_mask[0], token_type_ids[0], self.labels[idx]
        
    def __len__(self):
        return len(self.text_list)

class SentenceRankerPlus():
    def __init__(self, 
                train_set=None, 
                eval_set=None, 
                base_model_name = None, 
                model_name_or_path=None, 
                epochs = 4, 
                batch_size=32, 
                lr = 2e-5, 
                max_length=384, 
                mixed_precision=True,
                attention_window=None,
                device=None):
        logging.info("Trainer - initializing trainer")

        #setting pytorch parameters
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # assigning dataset
        self.train_set = train_set
        self.eval_set = eval_set

        self.model_name_or_path = model_name_or_path
        self.base_model_name = base_model_name
        
        #Fine-Tuning parameters
        if device is None:
            if torch.cuda.is_available():
                free_gpu = int(self.get_freer_gpu())
                logging.info(str(free_gpu))
                logging.info(type(free_gpu))
                torch.cuda.set_device(free_gpu)
                self.device = torch.device('cuda:' + str(free_gpu))
            else:
                torch.device('cpu')
                self.device('cpu')
        else:
            self.device = device
        

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.attention_window = attention_window
    
    def get_freer_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_eval_set(self, eval_set):
        self.eval_set = eval_set

    def load_model(self, base_model_name, model_name_or_path=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                free_gpu = int(self.get_freer_gpu())
                logging.info(str(free_gpu))
                logging.info(type(free_gpu))
                torch.cuda.set_device(free_gpu)
                self.device = torch.device('cuda:' + str(free_gpu))
            else:
                torch.device('cpu')
                self.device('cpu')
        else:
            self.device = device

        if model_name_or_path is None:
            if "longformer" in self.model_name_or_path:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels = 1, attention_window=64, torch_dtype=torch.float16)
                self.model.gradient_checkpointing_enable()
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels = 1)

        else:
            if "longformer" in model_name_or_path:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels = 1, torch_dtype=torch.float16)    
                self.model.gradient_checkpointing_enable()
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels = 1)    

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = self.model.to(self.device)

        
    
    def get_scores(self, sentences_list, abstract, batch_size):
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        abstracts = [abstract] * len(sentences_list)

        dataset = InternalDataset(text_list=sentences_list, 
                                context_list=abstracts, 
                                max_length=self.max_length,
                                tokenizer=self.tokenizer)
                                
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)
        torch.no_grad()
        scores = []

        self.model = self.model.to(self.device)
        self.model.eval()

        for step, batch in enumerate(dataloader):
            outputs = self.model(batch[0].to(self.device), 
                attention_mask = batch[1].to(self.device),)
            list_out = outputs[0].tolist()
            for out in list_out:
                scores.append(float(out[0]))
            torch.cuda.empty_cache()

        return scores

    # ----------------------------------------------------------------------------
    # Data Preparation
    # ----------------------------------------------------------------------------

    def prepare_training_data(self, label_keys):
        logging.info("Trainer - preparing training data")
        self.train_text = []
        self.train_abstract = []
        self.train_labels = []
        # the complete dictionary is self.dataset.train_set
        for k, d in tqdm(self.train_set.dataset.items()):
            try:
                if isinstance(d, tuple):
                    internal_dict = d[1]
                else:
                    internal_dict = d
                for s_k, s_t in internal_dict["raw_sentences"].items():
                    self.train_text.append(internal_dict["raw_sentences"][s_k]) #appending text
                    self.train_abstract.append(internal_dict["raw_abstract"]) #appending abstract
                    self.train_labels.append(internal_dict[label_keys][s_k])  #appending label Rouge-2 Precision - Length independent
            except Exception as e:
                print (e)
        
    def prepare_validation_data(self,label_keys):
        logging.info("Trainer - preparing validation data")
        self.val_text = []
        self.val_abstract = []
        self.val_labels = []
        # the complete dictionary is self.dataset.val_set
        for k, d in tqdm(self.eval_set.dataset.items()):
            try:
                if isinstance(d, tuple):
                    internal_dict = d[1]
                else:
                    internal_dict = d
                for s_k, s_t in internal_dict["raw_sentences"].items():
                    self.val_text.append(internal_dict["raw_sentences"][s_k]) #appending text
                    self.val_abstract.append(internal_dict["raw_abstract"]) #appending abstract
                    self.val_labels.append(internal_dict[label_keys][s_k])  #appending label Rouge-2 Precision - Length independent
            except Exception as e:
                print (e)

    def loading_pretrained_tokenizer(self, do_lower_case=False):
        logging.info("Trainer - loading pretrained tokenizer " + self.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, do_lower_case=do_lower_case)

    def prepare_for_training(self, label_keys="rlp_labels"):
        self.prepare_training_data(label_keys=label_keys)
        self.prepare_validation_data(label_keys=label_keys)
        self.loading_pretrained_tokenizer()
        self.prepare_dataloaders()

    def prepare_dataloaders(self):
        logging.info("Trainer - creating data loaders")
        self.tensor_train_labels = torch.tensor(self.train_labels) #set up labels
        self.tensor_val_labels = torch.tensor(self.val_labels) #set up labels

        self.train_dataset = InternalDataset(text_list=self.train_text,
                                            context_list=self.train_abstract,
                                            labels=self.tensor_train_labels,
                                            max_length=self.max_length,
                                            tokenizer=self.tokenizer)


        self.val_dataset = InternalDataset(text_list=self.val_text,
                                            context_list=self.val_abstract,
                                            labels=self.tensor_val_labels,
                                            max_length=self.max_length,
                                            tokenizer=self.tokenizer)

        # Create train and validation dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, 
            batch_size = self.batch_size, 
            shuffle = True, 
            num_workers=8)
        self.val_dataloader = DataLoader(self.val_dataset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers=8)

    def store_dataloaders(self, train_dl_path, val_dl_path):
        logging.info("Trainer - storing dataloaders")
        torch.save(self.train_dataloader, train_dl_path)
        torch.save(self.val_dataloader  , val_dl_path)

    def load_dataloaders(self, train_dl_path, val_dl_path):
        logging.info("Trainer - loading dataloaders")
        self.train_dataloader = torch.load(train_dl_path)
        self.val_dataloader = torch.load(val_dl_path)

    def fit(self, checkpoint_dir, print_freq=1000):

        logging.info("Trainer - preparing for fine-tuning")
        
        # Load the pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, 
            num_labels = 1, #for regression
            output_attentions = False, 
            output_hidden_states = False
        )

        if self.attention_window is not None:
            self.model.config.attention_window = self.attention_window

        self.model = self.model.to(self.device)
        # OR
        # self.model.cuda()
        logging.info("Current cuda device is: " + str(torch.cuda.current_device()))
        logging.info("self.device is : " + str(self.device))
        #logging.info(str(torch.cuda.memory_stats(device=None)))
        torch.cuda.empty_cache()

        # Create optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr = self.lr)
        total_steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        logging.info("Trainer - Starting fine-tuning")
        ##### Training #####
        for epoch in range(self.epochs):  
            self.model.train()
            total_loss, total_val_loss = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                self.model.zero_grad() 

                outputs = self.model(batch[0].to(self.device), 
                            attention_mask = batch[1].to(self.device), 
                            token_type_ids = batch[2].to(self.device), 
                            labels = batch[3].to(self.device))

                total_loss += outputs.loss.item()
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if step%print_freq == 0:
                    average_loss = outputs.loss.item()
                    logging.info("Epoch: " + str(epoch) + "\t Step: " + str(step) + "/" + str(len(self.train_dataloader)) + "\tTrain loss: " + str(average_loss))


            ##### Validation #####
            self.model.eval()
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                with torch.no_grad():  
                    outputs = self.model(batch[0].to(self.device), 
                                                attention_mask = batch[1].to(self.device), 
                                                token_type_ids = batch[2].to(self.device), 
                                                labels = batch[3].to(self.device))
                total_val_loss += outputs.loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)      
            avg_val_loss = total_val_loss / len(self.val_dataloader)     
            logging.info("Trainer - Train Loss     : " + str(avg_train_loss))
            logging.info("Trainer - Validation Loss: " + str(avg_val_loss))

            #saving as checkpoint
            epoch_name = epoch
            sanitized_model_name = self.base_model_name.replace("/", "-")
            ckp_dir = checkpoint_dir + "_" + sanitized_model_name + "_" + str(epoch_name)
            if not os.path.exists(ckp_dir):
                os.makedirs(ckp_dir)
            self.model.save_pretrained(ckp_dir)
        

    def continue_fit(self, checkpoint_dir, print_freq=1000, last_epoch=0, store_ckp_prefix=None):

        logging.info("Trainer - [CONTINUE] preparing for fine-tuning")
        if last_epoch != 0:
            logging.info("Resuming Epoch from last: " + str(last_epoch))
            logging.info("Model should be already loaded by using load_model()")
        
        logging.info("Current cuda device is: " + str(torch.cuda.current_device()))
        logging.info("self.device is : " + str(self.device))
        #logging.info(str(torch.cuda.memory_stats(device=None)))
        torch.cuda.empty_cache()

        # Create optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr = self.lr)
        total_steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        logging.info("Trainer - Starting fine-tuning")
        ##### Training #####
        for epoch in range(self.epochs):  
            epoch_name = last_epoch + epoch
            logging.info ("Storing at: " + str(store_ckp_prefix+str(epoch_name)))
            self.model.train()
            total_loss, total_val_loss = 0, 0
            for step, batch in enumerate(self.train_dataloader):
                self.model.zero_grad() 

                outputs = self.model(batch[0].to(self.device), 
                                            attention_mask = batch[1].to(self.device), 
                                            token_type_ids = batch[2].to(self.device), 
                                            labels = batch[3].to(self.device))

                total_loss += outputs.loss.item()
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if step%print_freq == 0:
                    average_loss = outputs.loss.item()
                    logging.info("Epoch: " + str(epoch) + "\t Step: " + str(step) + "/" + str(len(self.train_dataloader)) + "\tTrain loss: " + str(average_loss))


            ##### Validation #####
            self.model.eval()
            for i, batch in enumerate(tqdm(self.val_dataloader)):
                with torch.no_grad():  
                    outputs = self.model(batch[0].to(self.device), 
                                                attention_mask = batch[1].to(self.device), 
                                                token_type_ids = batch[2].to(self.device), 
                                                labels = batch[3].to(self.device))
                total_val_loss += outputs.loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)      
            avg_val_loss = total_val_loss / len(self.val_dataloader)     
            logging.info("Trainer - Train Loss     : " + str(avg_train_loss))
            logging.info("Trainer - Validation Loss: " + str(avg_val_loss))

            #saving as checkpoint
            epoch_name = last_epoch + epoch
            if not os.path.exists(store_ckp_prefix+str(epoch_name)):
                os.makedirs(store_ckp_prefix+str(epoch_name))
            self.model.save_pretrained(store_ckp_prefix+str(epoch_name))