import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
import sys 
import re
import torch.utils.data as D
from collections import defaultdict
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import takewhile
import json



class VideoCaptionDataset(D.Dataset):
    def __init__(self, video_features_path, captions_path):
        super(VideoCaptionDataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.captions_data = pd.read_json(captions_path)
        self.features_path = video_features_path
        self.vocabulary, self.max_caption_length = self._build_vocabulary()
        self.index_to_token, self.token_to_index = self._index_tokens()
        self.features, self.caption_info = self._prepare_dataset()

    def _build_vocabulary(self):
        vocabulary = Counter()
        max_length = 0
        for captions, _ in self.captions_data.itertuples(index=False):
            for caption in np.unique(captions):
                cleaned_caption = re.sub(r'[^\w\s]', '', caption).lower().split()
                vocabulary.update(cleaned_caption)
                max_length = max(max_length, len(cleaned_caption))
        vocabulary = Counter({word: count for word, count in vocabulary.items() if count > 3})
        max_length += 2  # Account for <BOS> and <EOS>
        return vocabulary, max_length

    def _index_tokens(self):
        special_tokens = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        word_to_index = {word: idx + 4 for idx, word in enumerate(self.vocabulary)}
        index_to_word = {idx + 4: word for idx, word in enumerate(self.vocabulary)}
        return {**special_tokens, **index_to_word}, {**{v: k for k, v in special_tokens.items()}, **word_to_index}

    def _prepare_dataset(self):
        video_features = {}
        captions_metadata = []
        for captions, video_id in self.captions_data.itertuples(index=False):
            video_features[video_id] = torch.from_numpy(np.load(f"{self.features_path}/feat/{video_id}.npy")).to(self.device)
            for caption in np.unique(captions):
                tokenized_caption = ["<BOS>"] + [word if word in self.vocabulary else "<UNK>" for word in re.sub(r'[^\w\s]', '', caption).lower().split()] + ["<EOS>"]
                tokenized_caption += ["<PAD>"] * (self.max_caption_length - len(tokenized_caption))
                token_ids = [self.token_to_index[word] for word in tokenized_caption]
                captions_metadata.append((caption, tokenized_caption, token_ids, video_id))
        return video_features, captions_metadata

    def __len__(self):
        return len(self.caption_info)

    def __getitem__(self, idx):
        original_caption, tokenized_caption, token_ids, video_id = self.caption_info[idx]
        token_tensor = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        one_hot_encoded = torch.nn.functional.one_hot(token_tensor, num_classes=len(self.index_to_token)).float()
        return self.features[video_id], one_hot_encoded, original_caption
    
class TestVideoDataset(D.Dataset):
    def __init__(self, videos_path, labels_path, vocabulary, word_to_token, token_to_index, max_length):
        super(TestVideoDataset, self).__init__()
        self.vocabulary = vocabulary
        self.word_to_token = word_to_token
        self.token_to_index = token_to_index
        self.max_length = max_length
        self.labels = pd.read_json(labels_path)
        self._prepare_dataset(videos_path)
        
    def _prepare_dataset(self, path):
        self.features = {}
        self.captions_info = []
        for entry in self.labels.itertuples(index=False):
            video_id = entry[1]
            self.features[video_id] = torch.from_numpy(np.load(f"{path}/feat/{video_id}.npy"))
            for caption in np.unique(np.array(entry[0])):
                processed_caption = ["<BOS>"]
                for word in re.sub(r'[^\w\s]', '', caption).lower().split():
                    processed_caption.append(word if word in self.vocabulary else "<UNK>")
                processed_caption.append("<EOS>")
                processed_caption += ["<PAD>"] * (self.max_length - len(processed_caption))
                tokenized_caption = [self.word_to_token[word] for word in processed_caption]
                
                # Index 0: Original Caption, Index 1: Processed Caption, Index 2: Tokenized Caption, Index 3: Video ID
                self.captions_info.append([caption, processed_caption, tokenized_caption, video_id])
                
    def __len__(self):
        return len(self.captions_info)
    
    def __getitem__(self, index):
        original_caption, _, tokenized_caption, video_id = self.captions_info[index]
        feature = self.features[video_id]
        caption_tensor = torch.Tensor(tokenized_caption)
        one_hot_encoded = torch.nn.functional.one_hot(caption_tensor.to(torch.int64), num_classes=len(self.token_to_index))
        return feature, one_hot_encoded, original_caption, video_id
    
    
    
    
class VideoToTextS2VT(nn.Module):
    def __init__(self, vocab_size, batch_size, frame_dimension, hidden_dimension, dropout_rate, frame_length, caption_length):
        super(VideoToTextS2VT, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.frame_dimension = frame_dimension
        self.hidden_dimension = hidden_dimension
        self.frame_length = frame_length
        self.caption_length = caption_length

        self.dropout = nn.Dropout(p=dropout_rate)
        self.feature_to_hidden = nn.Linear(frame_dimension, hidden_dimension)
        self.hidden_to_vocab = nn.Linear(hidden_dimension, vocab_size)

        self.encoder_lstm = nn.LSTM(hidden_dimension, hidden_dimension, batch_first=True)
        self.decoder_lstm = nn.LSTM(2 * hidden_dimension, hidden_dimension, batch_first=True)

        self.word_embedding = nn.Embedding(vocab_size, hidden_dimension)
        
    def forward(self, features, captions):
        features = self.feature_to_hidden(self.dropout(features.contiguous().view(-1, self.frame_dimension)))
        features = features.view(-1, self.frame_length, self.hidden_dimension)
        padding = torch.zeros(features.shape[0], self.caption_length - 1, self.hidden_dimension).to(self.device)
        features_padding = torch.cat((features, padding), dim=1)
        
        encoded_features, _ = self.encoder_lstm(features_padding)
        
        captions = self.word_embedding(captions[:, :self.caption_length - 1])
        caption_padding = torch.zeros(features.shape[0], self.frame_length, self.hidden_dimension).to(self.device)
        captions = torch.cat((caption_padding, captions), dim=1)
        captions = torch.cat((captions, encoded_features), dim=2)

        decoded_features, _ = self.decoder_lstm(captions)
        decoded_features = decoded_features[:, self.frame_length:, :]
        decoded_features = self.dropout(decoded_features.contiguous().view(-1, self.hidden_dimension))
        output = F.log_softmax(self.hidden_to_vocab(decoded_features), dim=1)
        return output
    
    def test(self, features):
        captions = []
        features = self.feature_to_hidden(self.dropout(features.contiguous().view(-1, self.frame_dimension)))
        features = features.view(-1, self.frame_length, self.hidden_dimension)
        padding = torch.zeros(features.shape[0], self.caption_length - 1, self.hidden_dimension).to(self.device)
        features = torch.cat((features, padding), dim=1)
        encoded_features, hidden_state = self.encoder_lstm(features)
        
        padding = torch.zeros(features.shape[0], self.caption_length - 1, self.hidden_dimension).to(self.device)
        caption_input = torch.cat((padding, encoded_features), dim=1)
        decoded_features, hidden_state = self.decoder_lstm(caption_input)
        
        bos_token = torch.ones(self.batch_size).to(self.device)
        caption_input = self.word_embedding(bos_token)
        caption_input = torch.cat((caption_input, encoded_features[:, 80, :]), dim=1).view(self.batch_size, 1, 2 * self.hidden_dimension)
        
        decoded_features, hidden_state = self.decoder_lstm(caption_input, hidden_state)
        decoded_features = torch.argmax(self.hidden_to_vocab(self.dropout(decoded_features.contiguous().view(-1, self.hidden_dimension))), dim=1)
        
        captions.append(decoded_features)
        for i in range(self.frame_length - 2):
            caption_input = self.word_embedding(decoded_features)
            caption_input = torch.cat((caption_input, encoded_features[:, self.frame_length + 1 + i, :]), dim=1)
            caption_input = caption_input.view(self.batch_size, 1, 2 * self.hidden_dimension)
            decoded_features, hidden_state = self.decoder_lstm(caption_input, hidden_state)
            decoded_features = decoded_features.contiguous().view(-1, self.hidden_dimension)
            decoded_features = torch.argmax(self.hidden_to_vocab(self.dropout(decoded_features)), dim=1)
            captions.append(decoded_features)
        return captions
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, optimizer, data_loader, batch_size, max_iterations, caption_size, vocabulary_size, total_epochs=1):
    model.to(device)
    loss_function = nn.NLLLoss()
    losses = []
    print(f"Training started for {total_epochs} epochs")
    for epoch in range(total_epochs):
        start_time = perf_counter()
        model.train()
        ground_truth_labels = []

        for iteration, batch in enumerate(data_loader):
            if iteration == max_iterations:
                break
            model.zero_grad()
            features = batch[0].requires_grad_().to(device)
            true_labels = batch[1].max(2)[1].to(device)
            output_labels = model(features.float(), true_labels)
            
            output_labels = output_labels.view(-1, caption_size-1, vocabulary_size)
            total_loss = 0
            for b in range(batch[0].shape[0]):          
                total_loss += loss_function(output_labels[b,:], true_labels[b,1:])
            total_loss.backward()
            optimizer.step()
            sys.stdout.write("\r")
            sys.stdout.write(f"[{'='*int(20*(iteration+1)/max_iterations)}] Epoch Completion {100*(iteration+1)/max_iterations}%")
            sys.stdout.flush()
        losses.append(total_loss.item()/len(data_loader))
        print(f" Epoch: {epoch}, Loss: {total_loss.item()/len(data_loader)}, Time Elapsed: {perf_counter()-start_time:.4f}s") 
    return losses

def evaluate_model(model, dataset, computation_device, caption_size, vocabulary_size, reverse_vocab, result_file):
    print("Evaluating Model")
    model.eval()
    model.to(device)
    loss_function = nn.NLLLoss()
    ground_truth_labels = []
    predicted_labels_list = []
    video_names = []
    with torch.no_grad():
        for index, batch in enumerate(dataset):
            model.zero_grad()
            features = batch[0].requires_grad_().to(computation_device)
            true_labels = batch[1].max(2)[1].to(computation_device)
            predictions = model(features.float(), true_labels)

            predictions = predictions.view(-1, caption_size-1, vocabulary_size)
            ground_truth_labels, predicted_labels_list = decode_predictions(predictions, batch[2], ground_truth_labels, predicted_labels_list, reverse_vocab)
            
            loss = 0
            for b in range(batch[0].shape[0]):          
                loss += loss_function(predictions[b,:], true_labels[b,1:])
                video_names.append(batch[3][b])
    write_to_file(video_names, predicted_labels_list, ground_truth_labels, output_filename=result_file)

def decode_predictions(predictions, labels, ground_truth, predicted, decoding_dict):
    end_tokens = ["<EOS>", "<PAD>"]
    prediction_indices = predictions.max(2)[1]
    for i in range(predictions.shape[0]):
        predicted_text = [decoding_dict[int(word_index.cpu().numpy())] for word_index in prediction_indices[i,:]]
        predicted_text = list(takewhile(lambda x: x not in end_tokens, predicted_text))
        
        ground_truth.append(str(labels[i]))
        predicted.append(" ".join(predicted_text))
    return ground_truth, predicted

def write_to_file(file_names, predicted_texts, true_labels, output_filename="results.txt"):
    with open(output_filename, "w") as file:
        for i in range(len(true_labels)):
            file.write(f"{file_names[i]}, {predicted_texts[i]}\n")
            
def calculate_average_bleu_score(predicted_captions_file="result.txt", reference_captions_json="./MLDS_hw2_1_data/testing_label.json"):
    # Load reference captions from the JSON file
    with open(reference_captions_json, 'r') as file:
        reference_captions_data = json.load(file)
    
    # Load predicted captions from the provided file
    predicted_captions = {}
    with open(predicted_captions_file, 'r') as file:
        for line in file:
            line = line.strip()
            separator_index = line.index(',')
            video_id = line[:separator_index]
            caption = line[separator_index + 1:]
            predicted_captions[video_id] = caption
    
    # Calculate BLEU scores for each video
    individual_bleu_scores = []
    for video in reference_captions_data:
        reference_captions = [caption.rstrip('.') for caption in video['caption']]
        bleu_score = BLEU(predicted_captions[video['id']], reference_captions, True)
        individual_bleu_scores.append(bleu_score)
    
    # Compute the average BLEU score across all videos
    average_bleu_score = sum(individual_bleu_scores) / len(individual_bleu_scores)
    print("Average BLEU score is:", average_bleu_score)
