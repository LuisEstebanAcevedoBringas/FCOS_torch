from create_dataset import VOCDataset, PascalVOCDataset,Dataset_G
from FCOS_utils import get_batch_statistics,ap_per_class
from transformation import get_transform
from others.utils import collate_fn
from pprint import PrettyPrinter
from tqdm import tqdm
import torch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = '/home/fp/Escritorio/Tandas/Mike/JSONFiles' #JSON del Mask dataset
batch_size = 16
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/FineTuning/Checkpoint_FT_Mask_epoca_25.pth.rar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

model.eval() # Switch to eval mode

# Load test data
# test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

#dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TEST', get_transform(True))

dataset = Dataset_G(data_folder, 'TEST', get_transform(True))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

def meanAP(data_loader_test, model):
    
    labels = []
    preds_adj_all = []
    annot_all = []

    for im, annot in tqdm(data_loader_test, position = 0, leave = True):
        im = list(img.to(device) for img in im)
        annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = make_prediction(model, im, 0.45)
            preds_adj = [{k: v.to(device) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
    
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.45) 
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(AP)
    print(f'mAP : {mAP}')
    print(f'AP : {AP}')

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def runEvaluate():
    meanAP(data_loader, model)