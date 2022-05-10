from fineTunedImageClassifier import *
from CustomDatasets import *



batch_size=32
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

num_classes = 17
model = ResnetImageEncoder(num_classes=num_classes)
model_path = "./saved_models/multi_label_image_classifier_resnet_fine_tuned.pth.tar"
model.load_state_dict(torch.load(model_path)['state_dict'])
model=model.cuda()

def getDataloaders(transform=None):
    dataset = UCM_Captions(transform=transform, ret_type="image-labels")
    UCM_train_set, UCM_test_set = torch.utils.data.random_split(dataset,
                                                                [int(dataset.__len__() * 0.8), dataset.__len__() -
                                                                 int(dataset.__len__() * 0.8)])
    TrainLoader = DataLoader(UCM_train_set, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(UCM_test_set, batch_size=batch_size, shuffle=True)

    return TrainLoader, TestLoader

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])
train_dataloader, test_dataloader = getDataloaders(preprocess)

def infer(model, dataloader, i=7):
    a, target = next(iter(dataloader))
    input = a[i]
    input = input.unsqueeze(0)
    input = input.to(device)
    #print(input)
    output = model(input)
    prediction = [0 if elem < 0.6 else 1 for elem in output[0]]
    print("Output")
    print(output)
    print("Predicted")
    print(prediction)
    print("Ground Truth")
    print(target[i])
    for index in range(17):
        if abs(prediction[index] - target[i][index]) < 0.01 and prediction[index] == 1:
            print(index)

#infer(model,train_dataloader)
def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def evalF1Score(model, dataloader, threshold=0.6):
    total_true_positives = 0
    total_target_positives = 0
    total_predicted_positives = 0
    precision = None
    recall = None
    F1score = None

    model.eval()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            scores = model(inputs)

            i = 2
            print("infer start")
            input_t = inputs[i]
            input_t = input_t.unsqueeze(0)
            input_t = input_t.to(device)
            # print(input)
            output = model(input_t)
            prediction = [0 if elem < 0.6 else 1 for elem in output[0]]
            print("Output")
            print(output)
            print("Predicted")
            print(prediction)
            print("Ground Truth")
            print(targets[i])
            for index in range(17):
                if abs(prediction[index] - targets[i][index]) < 0.01 and prediction[index] == 1:
                    print(index)
            print("infer end")


            # evaluating the predictions
            predictions = scores > threshold

            # evaluating parameters required for prcision and recall
            # precision = true_pos/(true_pos + false_pos) = true_pos/(total_pos_pred)
            # recall = true_pos/(true_pos + false_neg) = true_pos/(total_pos_targets)
            print("Target")
            print(targets[2])

            print("Predictions")
            print(predictions[2])


            predictions = predictions.flatten().float()
            targets = targets.flatten()
            total_target_positives_temp = 0
            total_target_positives_temp += torch.sum(targets == 1).float()
            total_target_positives += total_target_positives_temp

            total_predicted_positives_temp = 0
            total_predicted_positives_temp += torch.sum(predictions == 1).float()
            total_predicted_positives += total_predicted_positives_temp

            print(f"Correct F1 score: {f1_loss(targets, predictions)}")

            total_true_positives_temp = 0
            for elem in range(len(targets)):
                if (int(targets[elem]) == int(predictions[elem])) and (int(targets[elem]) == 1):
                    total_true_positives_temp += 1

            total_true_positives += total_true_positives_temp

            print("For batch")
            precision_temp = total_true_positives_temp / total_predicted_positives_temp
            recall_temp = total_true_positives_temp / total_target_positives_temp

            F1score_temp = (2 * precision_temp * recall_temp) / (precision_temp + recall_temp)
            print(f"Recall:{recall_temp:.4f}, Precision:{precision_temp:.4f}, F1score:{F1score_temp:.4f}")
        precision = total_true_positives / total_predicted_positives
        recall = total_true_positives / total_target_positives

        F1score = (2*precision * recall) / (precision + recall)

    model.train()

    print(f"Recall:{recall:.4f}, Precision:{precision:.4f}, F1score:{F1score:.4f}")

    return F1score


# evalF1Score(model, test_dataloader, threshold=0.6)