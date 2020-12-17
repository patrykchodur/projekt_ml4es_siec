import glob
import numpy as np
from PIL import Image

filenames = []

def load_images(directory):
    global filenames
    result = []
    for f in glob.iglob(directory + "/*"):
        img = np.asarray(Image.open(f))
        if not img.shape == (224, 224, 3):
            continue
        img = img / 255
        result.append(img)
        filenames.append(f)
    result = np.stack(result, axis=0)
    return result


def labels(images, label):
    result = np.zeros(images.shape[0], dtype=int) + label
    return result


def labels_matrix(labels_list, labels_no):
    result = np.zeros((labels_no, len(labels_list)))
    for i in range(len(labels_list)):
        result[labels_list[i]][i] = 1
    return result.T


# chinatree_images = load_images("data_downloader/chinatree_cut")
fig_images = load_images("fig_cut")
judastree_images = load_images("judastree_cut")
palm_images = load_images("palm_cut")
pine_images = load_images("pine_cut")

# chinatree_labels = labels(chinatree_images, 0)
fig_labels = labels(fig_images, 0)
judastree_labels = labels(judastree_images, 1)
palm_labels = labels(palm_images, 2)
pine_labels = labels(pine_images, 3)


all_images = np.concatenate((fig_images, judastree_images, palm_images, pine_images), 0)
all_labels = np.concatenate((fig_labels, judastree_labels, palm_labels, pine_labels), 0)

shuffler = np.random.permutation(all_images.shape[0])
all_images_shuffled = all_images[shuffler]
all_labels_shuffled = labels_matrix(all_labels[shuffler], 4)

#shuffle filenames
if len(filenames) != all_images.shape[0]:
    raise RuntimeError("filenames array has different length than all_images")
filenames_shuffled = [ "" for i in range(len(filenames)) ]
for i in range(len(filenames)):
    filenames_shuffled[shuffler[i]] = filenames[i]

split = 500
train_data = all_images_shuffled[split:]
train_labels = all_labels_shuffled[split:]
test_data = all_images_shuffled[:split]
test_labels = all_labels_shuffled[:split]

train_names = filenames_shuffled[split:]
test_names = filenames_shuffled[:split]

with open("train_names.txt", "w") as f:
    for item in train_names:
        f.write("%s\n" % item)

with open("test_names.txt", "w") as f:
    for item in test_names:
        f.write("%s\n" % item)


print('Data loaded')
