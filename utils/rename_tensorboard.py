import json
import os

class FileManager():
    def rename_files(path, generate_model_name, project_name):
        """Renames directories with tensorbord data.
        
        Takes in:
        - 'path' which should be the same as provided to tuner,
        - 'generate_model_name' function
            - should take in (iterable, **kwarg) and extract kwarg['hp'] which should correspond to hyperparameters
            - should produce unique, new name which will be displayed in tensorboard
            if model do not have any hyperparam values, property name with trial_id will be provided
        """
        path = f'{path}/{project_name}'
        folders = os.listdir(path)
        
        for f in folders:
            if(f.startswith('trial')):   
                with open(f'{path}/{f}/trial.json') as json_file:
                    data = json.load(json_file)
                    name = data['trial_id']
                    
                    hp = data['hyperparameters']['values']
                    if len(hp) == 0: hp = {'name' : name} 

                    new_name = generate_model_name(hp = hp)

                    head, _ = os.path.split(path)
                    try:
                        os.rename(f'{head}/{name}',f'{head}/{new_name}')
                    except:    
                        print(f'failed to rename {name} to {new_name}')