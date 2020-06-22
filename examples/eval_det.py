import os
import json
from datetime import datetime
import csv
from collections import OrderedDict

import api
from utils.evaluation import get_valset
from settings import PROJECT_ROOT

model_name = 'rapid'
weight_name = 'rapid_H1MW1024_Mar11_4000'
# val_set_names = ['Lunch2_mot', 'Edge_cases_mot', 'High_activity_mot',
#                  'All_off_mot', 'IRfilter_mot', 'IRill_mot']
# val_set_names = ['Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'High_activity',
#                  'All_off', 'IRfilter', 'IRill']
val_set_names = ['youtube_val']
input_size = 1024
conf_thres = 0.005

model_eval = api.Detector(
    model_name=model_name,
    weights_path=f'{PROJECT_ROOT}/weights/{weight_name}.pth'
)

csv_dic = OrderedDict()
csv_dic['weights'] = weight_name
csv_dic['inpu_size'] = input_size
csv_dic['tta'] = None
csv_dic['metric'] = 'AP_50'
csv_dic['date'] = datetime.now().strftime('%Y-%b-%d (%H:%M:%S)')

save_dir = f'./results/{weight_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for val_name in val_set_names:
    eval_info, val_func = get_valset(val_name)
    # dts_json = model_eval.eval_predict_vod(
    #     eval_info,
    #     input_size=input_size,
    #     conf_thres=conf_thres
    # )
    dts_json = model_eval.evaluation_predict(
        eval_info,
        input_size=input_size,
        conf_thres=conf_thres
    )
    # save results
    fname = f'{val_name}_{input_size}_{conf_thres}.json'
    save_path = os.path.join(save_dir, fname)
    json.dump(dts_json, open(save_path, 'w'), indent=1)

    eval_str, ap, ap50, ap75 = val_func(dts_json)
    print(eval_str)

    csv_dic[val_name] = ap50

with open('./results/results.csv', 'a', newline='') as csvfile:
    fields = [k for k in csv_dic.keys()]
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writerow({})
    writer.writeheader()
    writer.writerow(csv_dic)