import os
from torchvision import transforms
from torch.utils.data import DataLoader
from mydataset import Nutrition_RGBD

def get_DataLoader(args):
    train_transform = transforms.Compose([
                                # Resize the image to 320x448 pixels
                                transforms.Resize((320, 448)),
                                # Randomly flip the image horizontally for data augmentation
                                transforms.RandomHorizontalFlip(),
                                # Convert the image to a PyTorch tensor
                                transforms.ToTensor(),
                                # Normalize the tensor with mean and std for pre-trained models
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    test_transform = transforms.Compose([
                                # Resize the image to 320x448 pixels
                                transforms.Resize((320, 448)),
                                # Convert the image to a PyTorch tensor
                                transforms.ToTensor(),
                                # Normalize the tensor with mean and std for pre-trained models
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

    # get file paths
    nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
    nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgbd_train_processed.txt')
    nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgbd_test_processed1.txt')
    nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_train_processed.txt')
    nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_test_processed1.txt')

    # get training dataset
    trainset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt, transform = train_transform)
    # get testing dataset
    testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform = test_transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             ) 

    return train_loader, test_loader



