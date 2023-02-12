import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        #calculate f1_score
    if precision == 0 and recall == 0:
        f1_score= 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    #precision_recall.append((precision, recall))
    #print (precision)
    #print (recall)
    return precision, recall, f1_score, accuracy

# create a list of confidence thresholds to test
#thresholds = np.linspace(0, 1, 101)
thresholds = [0.5]
#thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


models = [
    {   'model_name': 'train123_l',
        'model_path': './runs/detect/Results_l/train123_l/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_research-2/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },

    {   'model_name': 'train12_l',
        'model_path': './runs/detect/Results_l/train12_l3/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_research-2/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },

    {   'model_name': 'train13_l',
        'model_path': './runs/detect/Results_l/train13_l/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_step1-1/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },

    {   'model_name': 'train1_l',
        'model_path': './runs/detect/Results_l/train1_l3/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_step1-1/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },

    {   'model_name': 'train03_l',
        'model_path': './runs/detect/Results_l/train03_l/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_step0-1/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },

    {   'model_name': 'train0_l',
        'model_path': './runs/detect/Results_l/train0_l/weights/best.pt',
        'data_path': '../Datasets/DX_uncleangear_step0-1/valid/images',
        'pattern': re.compile(r"/27.*ms"),
        'grundtruth': [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    },
    ]

#plt.figure(figsize=(10,10))
#results = []
# run yolo for each threshold and store the precision and recall values
for model in models:
    precision_list = []
    recall_list = []
    accuracy_list = []
    f1_score_list = []
    for threshold in thresholds:
        precision, recall, f1_score, accuracy = run_yolo(model['model_path'], threshold, model['data_path'], model['pattern'], model['grundtruth'])
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score)
    print (model['model_name'], "precision", precision_list, "recall", recall_list,"f1_score",f1_score_list, "accuracy",accuracy_list)    
    
    #results.append([model['model_name'], precision_list, recall_list, accuracy_list])
    
    '''
    plt.subplot(2,2,1)
    plt.plot( recall_list, precision_list, label=model['model_name'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()    
    plt.title('Precision-Recall curve')

    plt.subplot(2,2,2)
    plt.plot( thresholds, precision_list, label=model['model_name'])
    plt.xlabel('thresholds')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()    
    plt.title('Precision curve')

    plt.subplot(2,2,3)
    plt.plot( thresholds, recall_list, label=model['model_name'])
    plt.xlabel('thresholds')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()    
    plt.title('Recall curve')

    plt.subplot(2,2,4)
    plt.plot( thresholds, accuracy_list, label=model['model_name'])
    plt.xlabel('thresholds')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend()    
    plt.title('Accuracy curve')
    '''
#df = pd.DataFrame(results, columns=["Model Name", "Precision", "Recall", "Accuracy"])
#df.to_excel("results_uncleangear_imagelevel.xlsx", index=False)

#plt.show()

    


'''
# calculate the average precision and recall values for each threshold
#average_precision_recall = [(np.mean([pr[0] for pr in precision_recall_list]), np.mean([pr[1] for pr in precision_recall_list])) for precision_recall_list in precision_recall]

# plot the PR curve
#plt.plot(thresholds, average_precision_recall)
plt.plot( recall_list, precision_list,)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')

plt.show()

'''