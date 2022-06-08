from utils_ObjectDetection import get_batch_statistics,ap_per_class
from create_dataset import VOCDataset, PascalVOCDataset
from preferences.detect.utils import collate_fn
from transformation import get_transform
from evaluate.utils_map import *
from pprint import PrettyPrinter
from tqdm import tqdm

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = '/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 16
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '/home/fp/Escritorio/LuisBringas/FCOS/checkpoints/prueba_60_f.pth.rar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
# test_dataset = PascalVOCDataset(data_folder,
#                                 split='test',
#                                 keep_difficult=keep_difficult)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                           collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TEST', get_transform(True))

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=False, num_workers=0,
    collate_fn=collate_fn)

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
            preds_adj = make_prediction(model, im, 0.5)
            preds_adj = [{k: v.to(device) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
    
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5) 
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

#-----------
def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predictions = model(images)
            #print(predictions)
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = getBatchesImage(predictions, 'boxes'), getBatchesImage(predictions, 'labels'), getBatchesImage(predictions, 'scores')
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

#-------------
def runEvaluate():
    #evaluate(test_loader, model)
    meanAP(data_loader, model)

#-------------
def getBatchesImage(batch, item):
    out = []
    for i in batch:
        out.append(i[item])
    return out