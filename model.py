import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # setup network
        super(DecoderRNN, self).__init__()
        # setup embedding layer (maps input to embedding space)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # setup LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first = True, num_layers = num_layers)
        # setup dropout layer
        self.dropout = nn.Dropout(0.5)
        # setup linear layer to map hidden state to vocab size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # create embeddings from the captions of size (batch_size, max_len, embed_size)
        embeddings = self.embed(captions[:, :-1])
        # concatenate the embeddings and the features
        combined_input = torch.cat((features.unsqueeze(1), embeddings), 1)
        # pass the combined_input through the LSTM layer and output the hidden state
        lstm_out, _ = self.lstm(combined_input)
        # dropout the output of the LSTM layer
        lstm_out = self.dropout(lstm_out)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_word_indexes = []
        # output words only till max_len
        for i in range(max_len):
            # pass the inputs through the LSTM layer and output the hidden state
            lstm_out, states = self.lstm(inputs, states)
#             print("shape of lstm_out before squeeze ", lstm_out.shape)
            # pass through the linear layer
            outputs = self.linear(lstm_out.squeeze(1))
#             print("shape of lstm_out after squeeze ", lstm_out.squeeze(1).shape)
            # get the index of the highest probability word
            _, max_predicted_word_index = outputs.max(1)
            # shift tensor to cpu and get the value
            predicted_word_id = max_predicted_word_index.cpu().item()
            # append the predicted word index to the output list
            output_word_indexes.append(predicted_word_id)
            # break if the predicted word is the end of the sentence token
            if predicted_word_id == 1:
                break
            # get the embedding of the predicted word
            inputs = self.embed(max_predicted_word_index)
            # reshape the inputs to be of size (1, 1, embed_size) and pass through the LSTM layer for the next iteration
            inputs = inputs.unsqueeze(1)
#             print("shape of inputs ",inputs.shape)
        return output_word_indexes