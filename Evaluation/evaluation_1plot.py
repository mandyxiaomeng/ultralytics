import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

def run_yolo(model_path, conf_threshold, data_path, pattern, grundtruth):
    # run the yolo command with the given threshold
    result = subprocess.run(f'yolo task=detect mode=predict model={model_path} conf={conf_threshold} source={data_path} save=False workers=0', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print ("result", str(result))

    # parse the output to extract the precision and recall values for each image
    string = str(result.stderr)
    matches= re.findall(pattern, string)
    #print ('lines', len(matches))

    prediction=[]
    a=0

    for line in matches:
        #print (line)
        if "unclean," in line or "uncleans," in line:
            a = 1
        else:
            a = 0 
        prediction.append(a)
        
    gt=grundtruth
    pt=prediction
    #print(pt)
    #print(gt)

    correct_predictions = 0
    for i in range(len(gt)):
        if gt[i] == pt[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(gt)

    tp = sum([1 for i in range(len(pt)) if pt[i] == 1 and gt[i] == 1]) # count True Positives
    fp = sum([1 for i in range(len(pt)) if pt[i] == 1 and gt[i] == 0]) # count False Positives
    fn = sum([1 for i in range(len(pt)) if pt[i] == 0 and gt[i] == 1]) # count False Negatives
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0 # calculate precision
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0 # calculate recall
    #precision_recall.append((precision, recall))
    #print (precision)
    #print (recall)
    return precision, recall, accuracy

# create a list of confidence thresholds to test
#thresholds = np.linspace(0, 0.9, 3)
thresholds = 0.5
#thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model_name="train123_l"

model_path = "./runs/detect/Deburring/train123_l/weights/best.pt"

data_path = "../Datasets/Deburring/DX_deburring_step12-v2/valid/images"  

#pattern = re.compile(r"/27.*ms")
pattern = re.compile(r"/32.*ms")

#grundtruth= [0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,0,0]
grundtruth= [1,0,1,0,1,0,1,0,0]
# run yolo for each threshold and store the precision and recall values
precision_list = []
recall_list = []
accuracy_list = []
for threshold in thresholds:
    precision, recall, accuracy = run_yolo(model_path, threshold, data_path, pattern, grundtruth)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)
    
print (model_name, "precision", precision_list, "recall", recall_list, "accuracy", accuracy_list)    

# calculate the average precision and recall values for each threshold
#average_precision_recall = [(np.mean([pr[0] for pr in precision_recall_list]), np.mean([pr[1] for pr in precision_recall_list])) for precision_recall_list in precision_recall]
#average_precision = np.mean(precision_list)
#average_recall = np.mean(recall_list)
#average_accuracy = np.mean(accuracy_list)

'''
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
# plot the PR curve
#plt.plot(thresholds, average_precision_recall)
#plt.plot( recall_list, precision_list,)
plt.plot(thresholds, precision_list, label = "precision")
plt.plot(thresholds, recall_list, label = "recall")
plt.plot(thresholds, accuracy_list, label = "accuracy")
#plt.xlabel('Recall')
#plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.legend()
plt.title('threhold curve')



plt.subplot(1,2,2)
# plot the PR curve
#plt.plot(thresholds, average_precision_recall)
plt.plot( recall_list, precision_list,)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.legend()
plt.title('Precision-Recall curve')
plt.show()
'''