import os, sys
import pandas as pd

dirname = os.path.join(os.path.dirname( __file__ ), os.pardir)
class DataOperations():
    max_int = 999999
    
    def get_data(dir, skip = 0, take = max_int):
        df = None
        for i in os.listdir(dir):
            sub_dir = os.path.join(dir,i)
            if(df is None): 
                df = DataOperations.get_data_for_category(sub_dir, skip=skip, take=take)
            else:
                df = df.append(DataOperations.get_data_for_category(sub_dir, skip=skip, take=take))

        return df.sample(frac=1).reset_index(drop=True)

    def get_data_for_category(dir, skip = 0, take = max_int):
        a=[]
        _, label = os.path.split(dir)
        for k,j in enumerate(os.listdir(dir)):
            if k>=take:continue
            if k<skip:continue
            a.append((f'{dir}/{j}',label))

        cat_df = pd.DataFrame(a,columns=['filename','class'])
        return cat_df

    def move_files(source, dest):
        source_path = os.path.join(dirname, source)
        for i in os.listdir(source_path):
            i_path = os.path.join(dirname, dest, i) 

            if not os.path.exists(i_path):
                os.makedirs(i_path)
            for k,j in enumerate(os.listdir(os.path.join(source_path, i))):
                if j.startswith('Private'): continue

                os.replace(os.path.join(dirname, source, i, j), os.path.join(i_path, j))