

import pandas as pd


def extract(interest, df, subdata=False):
    
    temp_df = df[interest]
    temp_df = temp_df.sum(axis=1)
    if subdata:
        temp_df = temp_df.replace(0, 10)
        temp_df = temp_df.replace([1,2,3,4,5,6], 0)
        temp_df = temp_df.replace(10, 1)
        
    else:
        temp_df = temp_df.replace([2,3,4,5,6], 1)
    
    return temp_df.tolist()


def tain_val_csv(df, name):
    """Converts dataframe into csv files"""
    
    np.random.seed(12)
    random_idx = np.random.choice(df.shape[0], size=df.shape[0], replace=False)
    
    val_idx = random_idx[:int(df.shape[0]*0.15)]
    val_df = df.iloc[val_idx]
    df.to_csv(('data/dataframe/' + name + '_val.csv'))
    
    train_idx = random_idx[int(df.shape[0]*0.15):]
    train_df = df.iloc[train_idx]
    df.to_csv(('data/dataframe/' + name + '_train.csv'))


def main():

    nothing_list = extract(['No Finding', 'Support Devices', 'Fracture'], train_df)
    lungs_list = extract(['Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Lung Lesion', 'Atelectasis'], train_df)
    cardio_list = extract(['Enlarged Cardiomediastinum', 'Cardiomegaly'], train_df)
    pleural_list = extract(['Pneumothorax', 'Pleural Other', 'Pleural Effusion'], train_df)
    
    # main dataframe
    main_train_df = pd.DataFrame({'Path': train_df['Path'].tolist()})
    main_train_df['nothing'] = nothing_list
    main_train_df['lungs'] = lungs_list
    main_train_df['cardio'] = cardio_list
    main_train_df['pleural'] = pleural_list

    # nothing dataframe
    temp_ls = ['Path', 'No Finding', 'Support Devices', 'Fracture']
    nothing_train_df =  train_df[temp_ls]
    nothing_train_df['none'] = extract(temp_ls, train_df, subdata=True)

    # lungs dataframe
    temp_ls = ['Path', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Lung Lesion', 'Atelectasis']
    lungs_train_df =  train_df[temp_ls]
    lungs_train_df['none'] = extract(temp_ls, train_df, subdata=True)

    # cardio dataframe
    temp_ls = ['Path', 'Enlarged Cardiomediastinum', 'Cardiomegaly']
    cardio_train_df =  train_df[temp_ls]
    cardio_train_df['none'] = extract(temp_ls, train_df, subdata=True)
                            
    # pleural dataframe
    temp_ls = ['Path', 'Pneumothorax', 'Pleural Other', 'Pleural Effusion']
    pleural_train_df =  train_df[temp_ls]
    pleural_train_df['none'] = extract(temp_ls, train_df, subdata=True)

    # Saves files as .csv
    tain_val_csv(main_train_df, name='main')
    tain_val_csv(cardio_train_df, name='cardio')
    tain_val_csv(nothing_train_df, name='nothing')
    tain_val_csv(pleural_train_df, name='pleural')
    tain_val_csv(lungs_train_df, name='lungs')



if __name__ =='__main__':

    train_df = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
    train_df = train_df.fillna(0)
    train_df = train_df.replace(-1, 1)

    main()