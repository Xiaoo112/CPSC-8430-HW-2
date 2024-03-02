import torch
from bleu_eval import *
from utils import *


data_loc = sys.argv[1]
output_filename = sys.argv[2]

training_data_path = f'{data_loc}/MLDS_hw2_1_data/training_data'
training_labels = f'{data_loc}/MLDS_hw2_1_data/training_label.json'

ds = VideoCaptionDataset(training_data_path, training_labels) # Length of caption: 40+2 for bos, eos , items in vocab=2406


num_hidden = 256
num_vocab = len(ds.index_to_token)
detokenize_dict = ds.index_to_token
num_feat = 4096
batch_size = 64
caption_len = ds.max_caption_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drop = 0.3
lr=1e-4
iters = len(ds) // batch_size
epochs = 100


mod1 = VideoToTextS2VT(num_vocab, batch_size, num_feat, num_hidden, drop, 80, caption_len)
dataset = D.DataLoader(ds, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(mod1.parameters(), lr=lr)

print(f"Training for {batch_size} bs, {epochs} epochs, {lr} lr, {num_hidden} hid dim")
losses = train_model(model=mod1, 
    optimizer=optimizer, 
    data_loader=dataset,  
    max_iterations=iters, 
    caption_size=caption_len, 
    batch_size = batch_size,
    vocabulary_size=num_vocab, 
    total_epochs=epochs
)


torch.save(mod1.state_dict(), f'model.pth')


figure = plt.figure(figsize=(8, 6))
plt.plot(losses, color = "orange")
plt.title('Loss Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show() 
figure.savefig('Loss Function.png')
torch.manual_seed(0)



#output_filename = "/home/xiaofey/CPSC-8430/Homework#2/result.txt"

test_data_path = f"{data_loc}/MLDS_hw2_1_data/testing_data/"
test_labels = str(f"{data_loc}/MLDS_hw2_1_data/testing_label.json")
test_ds = TestVideoDataset(test_data_path, test_labels, ds.vocabulary, ds.token_to_index, ds.index_to_token, ds.max_caption_length)

batch_size = 10
hidden_dim = 256
vocab_size = len(ds.index_to_token)
detokenize_dict = ds.index_to_token
feat_size = 4096
seq_length = 80
caption_length = ds.max_caption_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
drop = 0.3
lr=1e-4

its = len(ds) // batch_size


mod2 = VideoToTextS2VT(vocab_size, batch_size, feat_size, hidden_dim, drop, 80, caption_length)
test_dataset = D.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
opt = torch.optim.Adam(mod2.parameters(), lr=lr)


mod2.load_state_dict(torch.load("./model.pth"))
evaluate_model(mod2, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename)


calculate_average_bleu_score(predicted_captions_file=output_filename, reference_captions_json=test_labels)