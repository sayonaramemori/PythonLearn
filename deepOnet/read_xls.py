import pandas
import torch

def get_data():
    df = pandas.read_excel('./datas.xlsx')
    # print(df)
    for i in range(len(df)):
        temp = df.iloc[i,:]
        sl = [i for i in temp]
        binput = torch.tensor(sl[:3])
        tinput = torch.tensor(sl[3:4])
        label = torch.tensor(sl[-1:])
        # print(label)
        yield (binput,tinput,label)

def get_data_csv():
    df = pandas.read_csv('./flattened_data.csv')
    print(df)
    for i in range(len(df)):
        temp = df.iloc[i,:]
        sl = [i for i in temp]
        binput = torch.tensor(sl[:3])
        tinput = torch.tensor(sl[3:4])
        label = torch.tensor(sl[4:5])
        # print(label)
        yield (binput,tinput,label)


if __name__ == "__main__":
    temp = get_data_csv()
    for i in temp:
        print(i)
