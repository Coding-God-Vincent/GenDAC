import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

'''注意事項
* 要先把會用到的資料夾創建好
* 本檔案是將 tensorboard 的 event 轉成 csv 檔後再進一步讀取 csv 檔的內容後做成圖
'''

# generate EventGenerator (reloaded) and return
# event_path : event 的位址
def generate_EA(event_path):
    EA = EventAccumulator(
        path= event_path,
        # '0' 代表要全部
        size_guidance= {
            'tensors' : 0,
            'scalars' : 0
        }
    )
    EA.Reload()  # 讀取檔案
    return EA


# 回傳 file_path / algo_name.csv (instance of Path)
# file_path : 要存所有 .csv 檔的位址 (還沒分演算法)
def generate_original_csv_path(algo_name, file_path):
    algo_csv = algo_name + '_csv'
    return file_path / algo_csv


# generate csv files to the target path
# EA : generate_EA() 回傳的 EA
# original_csv_path : 要儲存 csv 檔案的地方 (Path)
# target_tags tensorboard 中的變數名稱, ex : ['qoe/volte', 'qoe/embb_general', 'qoe/urllc', 'se', 'reward', 'utility']
def generate_csv(EA, original_csv_path, target_tags):
    for tag_name in target_tags:
        csv_path = original_csv_path
        event = EA.Scalars(tag= tag_name)
        df = pd.DataFrame([(e.step, e.value) for e in event], columns= ['Step', 'Value'])
        file_name = tag_name.replace("/", "_") + '.csv'  # 將 tag_name 中有 "/" 的改為 "_"，後面加一個 .csv
        csv_path = csv_path / file_name
        df.to_csv(str(csv_path), index= False)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 1. 取出各演算法各指標的 csv 檔到指定位址

'''hyperparameters'''
# 要取出的指標的值
target_tags = [
    'qoe/volte', 'qoe/embb_general', 'qoe/urllc', 
    'se', 'reward', 'utility', 
    'observationBits/volte', 'observationBits/embb_general', 'observationBits/urllc',
    'individual_se/embb_general', 'individual_se/volte', 'individual_se/urllc',
    'action/embb_general', 'action/volte', 'action/urllc',
    # 'idle_frame'
]
# target_tags = ['qoe/volte', 'qoe/embb_general', 'qoe/urllc', 'se', 'utility']


'''*****algo1 : GenDAC'''
# # 想將 tensorboard 中的 event 內容轉成各個 csv 檔後放在 : /home/super_trumpet/NCKU/Paper/My Methodology/Outcome/Combine/D2AC_csv 中
# # 例如 utility.csv 會放在 /home/super_trumpet/NCKU/Paper/My Methodology/Outcome/Combine/D2AC_csv/utility.csv
# algo_name = 'GenDAC_DDPM_5'
# # # file_path : 要存所有 .csv 檔的位址 (還沒依照算法分資料夾) (Instance of Path)
# file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
# event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/GenDAC/exp21/tensorboard/events.out.tfevents.1769329654.SuperTrumpet.5235.0'
# # 按照各指標將結果存入各種不同的 csv 檔
# generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)




'''*****algo2 : GANDDQN'''
# algo_name = 'GANDDQN'
# file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
# event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/GANDDQN/exp7/tensorboard/events.out.tfevents.1769439467.SuperTrumpet.131409.0'
# generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)


'''****algo3 : LSTM_A2C'''
# algo_name = 'LSTM_A2C'
# file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
# event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/LSTM_A2C/exp4/tensorboard/events.out.tfevents.1769498585.SuperTrumpet.355707.0'
# generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)


# '''****algo4 : Hard-Slicing (沒有紀錄 reward，記得把 tags 中的 reward 弄掉)''' 
# algo_name = 'Hard_Slicing'
# file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
# event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/Hard_Slicing/exp4/tensorboard/events.out.tfevents.1769490491.SuperTrumpet.314231.0'
# generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)

'''****algo5 : SAC'''
# algo_name = 'SAC'
# file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
# event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/SAC/exp5/tensorboard/events.out.tfevents.1769523551.SuperTrumpet.451036.0'
# generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)

'''****algo6 : PPO'''
algo_name = 'PPO'
file_path = Path('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine')
event_path = '/home/super_trumpet/NCKU/Paper/My Methodology/Logs/Logs_fixedUE_env/PPO/exp4/tensorboard/events.out.tfevents.1769504617.SuperTrumpet.384040.0'
generate_csv(EA= generate_EA(event_path= event_path), original_csv_path= generate_original_csv_path(algo_name= algo_name, file_path= file_path), target_tags= target_tags)