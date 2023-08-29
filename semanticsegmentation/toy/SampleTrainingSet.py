import os
import shutil
import math
import random
import pickle


def sample_train(path_data, percentage, subfolders=['gt', 'img'], seed=5):
    random.seed(seed)
    train_path_data = os.path.join(path_data, '/train/')
    target_path = os.path.join(path_data + '_' + str(math.floor(percentage * 100), '/train/'))

    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for i in subfolders:
        if not os.path.isdir(os.path.join(target_path, i)):
            os.makedirs(os.path.join(target_path, i))

    try:
        with open('./all_files_' + math.floor(percentage * 100) + '.pickle', 'rb') as handle:
            files_extracted = pickle.load(handle)
        for current_file in files_extracted:
            for i in subfolders:
                shutil.copy(os.path.join(path, i, current_file), os.path.join(target_path, i, current_file))
    except FileNotFoundError:
        all_files = os.listdir(os.path.join(train_path_data, subfolders[0]))

        total_number_files = len(all_files)
        number_files_to_extract = math.floor(total_number_files * percentage)

        files_extracted = random.sample(all_files, number_files_to_extract)

        for current_file in files_extracted:
            for i in subfolders:
                shutil.copy(os.path.join(path, i, current_file), os.path.join(target_path, i, current_file))

    return


def copy_val(path_data, percentage, subfolders=['gt', 'img']):
    val_path_data = os.path.join(path_data, '/val/')
    target_path = os.path.join(path_data + '_' + str(math.floor(percentage * 100), '/val/'))

    all_files = os.listdir(os.path.join(val_path_data, subfolders[0]))

    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for i in subfolders:
        if not os.path.isdir(os.path.join(target_path, i)):
            os.makedirs(os.path.join(target_path, i))

    for current_file in all_files:
        for i in subfolders:
            shutil.copy(os.path.join(path, i, current_file), os.path.join(target_path, i, current_file))
    return


if __name__ == "__main__":
    # Adjust the seed value for different samples if necessary. The seed will not generate the same files as provided in the .pickle files in the
    # repository.
    sample_train(path_data='./data/toy', percentage=0.1, subfolders=['gt', 'img'], seed=5)
    copy_val(path_data='./data/toy', percentage=0.1, subfolders=['gt', 'img'])
    sample_train(path_data='./data/toy', percentage=0.5, subfolders=['gt', 'img'], seed=5)
    copy_val(path_data='./data/toy', percentage=0.5, subfolders=['gt', 'img'])
