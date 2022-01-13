import matplotlib.pyplot as plt
import pandas as pd 
import wandb

def extract_data(project_author, project_name):
    api = wandb.Api()
    entity, project = project_author, project_name
    runs = api.runs(entity + "/" + project) 
    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})
        name_list.append(run.name)

    return pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

