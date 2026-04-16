from utils import dataset_new
distribution=["bia"]
def choose(args):
    ls = args.dataset.split('_')
    if ls[-1] not in distribution:
        distr=""
        dataname=args.dataset
    else:
        distr = ls[-1]
        dataname="_".join(ls[:-1])
    file_name = []
    if dataname == 'scene':
        print('Data Preparation of scene')
        file_name = ["./data/scene/scene_data.csv", "./data/scene/scene_label.csv", "./data/scene/scene_com_label.csv"]
        num_class = 6
        input_dim = 294
    elif dataname == "yeast":
        print('Data Preparation of yeast')
        file_name = ["./data/yeast/yeast_data.csv", "./data/yeast/yeast_label.csv", "./data/yeast/yeast_com_label.csv"]
        num_class = 14
        input_dim = 103
    if distr=="":
        train_loader, test_loader = dataset_new.ComFold(args.batch_size, file_name, 10, args.fold)
        print("uniform distribution")
        return train_loader, test_loader, num_class, input_dim
    elif distr=="bia":
        biased_file=file_name[-1].split("_com_label.csv")[0]
        biased_file =biased_file+"_com_bia_label.csv"
        file_name[-1] = biased_file
        train_loader, test_loader = dataset_new.ComFold(args.batch_size, file_name, 10, args.fold)
        print("biased distribution")
        return train_loader, test_loader, num_class, input_dim