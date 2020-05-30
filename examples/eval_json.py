import os
import json
from datetime import datetime
import csv
from collections import OrderedDict

import api
from utils.evaluation import get_valset
from settings import COSSY_DIR, PROJECT_ROOT


dts_path = f'{PROJECT_ROOT}/results/tmp/Edge_cases_mot_1024_0.005.json'

valset_name = 'Edge_cases'
gt_path = f'{COSSY_DIR}/annotations/{valset_name}.json'


# csv_dic = OrderedDict()
# csv_dic['weights'] = weight_name
# csv_dic['inpu_size'] = input_size
# csv_dic['tta'] = None
# csv_dic['metric'] = 'AP_50'
# csv_dic['date'] = datetime.now().strftime('%Y-%b-%d (%H:%M:%S)')


dts_json = json.load(open(dts_path, 'r'))

eval_info, val_func = get_valset(valset_name)
eval_str, ap, ap50, ap75 = val_func(dts_json)
print(eval_str)

# csv_dic[val_name] = ap50

# with open('./results/results.csv', 'a', newline='') as csvfile:
#     fields = [k for k in csv_dic.keys()]
#     writer = csv.DictWriter(csvfile, fieldnames=fields)
#     writer.writerow({})
#     writer.writeheader()
#     writer.writerow(csv_dic)