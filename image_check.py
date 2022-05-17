import torch

from fineTunedImageClassifier import *
from CustomDatasets import *
import cv2 as cv

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
    model.eval()
    with torch.no_grad():
        a, target = next(iter(dataloader))
        input = a[i]
        input_img = input.detach().permute(1, 2, 0).numpy()
        input_img = input_img + 0.5
        input = input.unsqueeze(0)
        input = input.to(device)
        #print(input)
        output = model(input)
        cv.imshow("Input Image", input_img)
        cv.imwrite('./images/input_image.jpg', input_img)
        output = output.to('cpu').numpy()
        gt=target[i].to('cpu').numpy()
        prediction = [0 if elem < 0.6 else 1 for elem in output[0]]
        print("Output")
        print(output)

        print("Predicted")
        print(prediction)
        print("Ground Truth")
        print(gt)
        # for index in range(17):
        #     if abs(prediction[index] - target[i][index]) < 0.01 and prediction[index] == 1:
        #         print(index)
        #cv.imwrite('./images/input_image.jpg',input_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return output,gt

output,gt=infer(model,train_dataloader)

def evalF1Score(model, dataloader, threshold=0.6,i=7):
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

            # evaluating the predictions
            predictions = scores > threshold


            for i in range(len(targets)):
                input = inputs[i]
                input = input.unsqueeze(0)
                input = input.to(device)
                # print(input)
                output = model(input)
                prediction = [0 if elem < 0.6 else 1 for elem in output[0]]
                # print("---------------------")
                print("--------------------------Output")
                print(output.to('cpu').numpy())
                # print("--------------------Predicted")
                # print(torch.tensor(prediction,dtype=torch.float32,device=device, requires_grad=True))
                print("--------------------Ground Truth")
                print(targets[i].to('cpu').numpy())
            # evaluating parameters required for prcision and recall
            # precision = true_pos/(true_pos + false_pos) = true_pos/(total_pos_pred)
            # recall = true_pos/(true_pos + false_neg) = true_pos/(total_pos_targets)

            # print("-------------------------")
            # print(predictions)
            # print(targets)
            # print(targets[2])
            # print(predictions[2])
            predictions = predictions.flatten().float()
            targets = targets.flatten()
            # print("-------------------------")
            # print(predictions)
            # print(targets)


            total_target_positives += torch.sum(targets == 1).float()
            total_predicted_positives += torch.sum(predictions == 1).float()

            for elem in range(len(targets)):
                if (int(targets[elem]) == int(predictions[elem])) and (int(targets[elem]) == 1):
                    total_true_positives += 1

        precision = total_true_positives / total_predicted_positives
        recall = total_true_positives / total_target_positives

        F1score = (2*precision * recall) / (precision + recall)

    model.train()

    print(f"Recall:{recall:.4f}, Precision:{precision:.4f}, F1score:{F1score:.4f}")

    return F1score


#evalF1Score(model, test_dataloader, threshold=0.6)

