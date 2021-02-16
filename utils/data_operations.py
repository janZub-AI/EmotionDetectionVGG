import os, sys
import pandas as pd

class DataOperations():
    max_int = sys.maxsize * 2 + 1

    def get_data(dir, skip = 0, take = max_int):
        print(dir)
        df = None
        for i in os.listdir(dir):
            print(i)
            take = take // len(os.listdir(dir))
            sub_dir = os.path.join(dir,i)
            if(df is None): 
                df = DataOperations.get_data_for_category(sub_dir, skip=skip, take= take)
            else:
                df = pd.concat([df, DataOperations.get_data_for_category(sub_dir, skip=skip, take= take)])

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