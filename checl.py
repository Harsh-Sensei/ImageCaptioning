from os import getcwd, chdir
from embeddingText_En_De import *
from CustomDatasets import *
#from torchtext.data.metrics import bleu_score

torch.manual_seed(73)

torch.autograd.set_detect_anomaly(True)

feature_dim = 1000
en_num_layers = 2
de_num_layers = 2
en_hidden_size = 1024
de_hidden_size = 1024
embedding_dim = 300
vocab_size = 309 #to be defined after datasets are loaded
UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getTextUCMDataLoader(batch_size=batch_size)


def getTextEncoder(model_path="./saved_models/Embed_e_LSTM_d_LSTM_UCM.pth.tar"):
    model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                        en_hidden_size=en_hidden_size,
                                        num_layers=num_layers,
                                        vocab_size=vocab_size,
                                        de_hidden_size=de_hidden_size,
                                        pad_idx=pad_idx,
                                        p=dropout_p,
                                        teacher_force_ratio=teacher_force_ratio).to(device=device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.encoder, model

_,model=getTextEncoder()

def test(dataloader, model, device, i=3):
    input, _ = next(iter(dataloader))
    input = input[1]
    input = input.unsqueeze(0).to(device)
    output = model.inference(input)
    output = output.squeeze(0)
    output = output.argmax(1)

    print(input)
    print(output)

    return None

test(UCM_train_loader, model, device)


def tostring(vectr,dicti):
    s = []
    string = []
    for v in vectr:
        s.append(v.item())
        if dicti[v.item()] == "<EOS>":
            break
        string.append(dicti[v.item()])
    return string[1:]

def testing(dataloader, model, device, i=3):
    input, _ = next(iter(dataloader))
    input = input[1]
    strs = []
    for d in data:
        strs.append([tostring(d)])
    data = data.to(device=device)
    output = model(data)
    output = output.argmax(2)
    output_str = []
    for o in output:
        output_str.append(tostring(o))
    score = bleu_score(output_str, strs)
    print(score)
    return score
