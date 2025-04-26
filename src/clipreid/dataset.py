from clipreid.utils import read_image, download_url, mkdir_if_missing
import glob
import os
import os.path as osp
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import zipfile


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    # Set to True for datasets with test sets having hidden/private identity labels.
    # Resulting query to gallery ranking result should be exported for external evaluation with private identity labels.
    hidden_labels = False

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train) != 0 and len(train[0]) == 3:
            train = [(*items, 0) for items in train]
        if len(query) != 0 and len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery) != 0 and len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid, dsetid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            dsetid += self.num_datasets
            train.append((img_path, pid, camid, dsetid))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        if self._train_only:
            return

        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid, dsetid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid, dsetid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class Soccernetv3(ImageDataset):
    """Soccernet-v3 train and valid sets. When set as "source" in the run configs (cfg.data.sources), the train set is
    used for training. When set as "target" set in the run configs (cfg.data.targets), the valid set is used for performance
    evaluation.
    """
    dataset_dir = 'raw_dataset/soccernet-reid/raw'

    def __init__(self, root='', soccernetv3_training_subset=1.0, **kwargs):
        assert 1.0 >= soccernetv3_training_subset > 0.0

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.reid_dataset_dir = self.download_soccernet_dataset(
            self.dataset_dir, ["valid", "train"])

        self.train_dir = osp.join(self.reid_dataset_dir, 'train')
        self.query_dir = osp.join(self.reid_dataset_dir, 'valid/query')
        self.gallery_dir = osp.join(self.reid_dataset_dir, 'valid/gallery')

        required_files = [
            self.reid_dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train, _, _ = self.process_dir(self.train_dir, {
        }, relabel=True, soccernetv3_training_subset=soccernetv3_training_subset)

        query, pid2label, ids_counter = self.process_dir(self.query_dir, {}, 0)
        gallery, pid2label, ids_counter = self.process_dir(
            self.gallery_dir, pid2label, ids_counter)

        super(Soccernetv3, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, main_path, pid2label, ids_counter=0, relabel=False, soccernetv3_training_subset=1.):
        data = []
        img_paths = glob.glob(osp.join(main_path, '*/*/*/*/*.png'))
        # sort images list such that each sample position in the list match its filename index
        img_paths.sort(key=lambda img_path: self.get_bbox_index(img_path))

        # if soccernetv3_training_subset is set, use samples from action '0' to action 'end_action'
        action_num = self.extract_sample_info(
            os.path.basename(img_paths[-1]))["action_idx"] + 1
        end_action = action_num * soccernetv3_training_subset

        for img_path in img_paths:
            filename = os.path.basename(img_path)
            info = self.extract_sample_info(filename)
            pid = info["person_uid"]
            action_idx = info["action_idx"]
            if action_idx >= end_action:
                break
            if relabel:
                if pid not in pid2label:
                    pid2label[pid] = ids_counter
                    ids_counter += 1
                pid = pid2label[pid]
            data.append((img_path, pid, action_idx))

        return data, pid2label, ids_counter

    @staticmethod
    def download_soccernet_dataset(dataset_dir, split):
        task = "reid"
        reid_dataset_dir = osp.join(dataset_dir, task)

        mySNdl = SNdl(LocalDirectory=dataset_dir)

        for set_type in split:
            # download SoccerNet dataset subsets specified by 'set_type' (train/valid/test/challenge)
            path_to_set = osp.join(reid_dataset_dir, set_type)
            if osp.exists(path_to_set):
                print("SoccerNet {} set was already downloaded and unzipped at {}.".format(
                    set_type, path_to_set))
                continue

            mySNdl.downloadDataTask(task=task, split=[set_type])

            print("Unzipping {} set to '{}' ...".format(
                set_type, reid_dataset_dir))
            path_to_zip_file = osp.join(reid_dataset_dir, set_type + ".zip")
            if not osp.exists(path_to_zip_file):
                raise FileNotFoundError(
                    "Missing zip file {}.".format(path_to_zip_file))
            else:
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    zip_ref.extractall(reid_dataset_dir)
            print("Deleting {} set zip file at '{}'...".format(
                set_type, path_to_zip_file))
            os.remove(path_to_zip_file)

            print('SoccerNet {} set is ready.'.format(set_type))

        return reid_dataset_dir

    @staticmethod
    def extract_sample_info(filename):
        """ Extract sample annotations from its filename
            File naming convention is:
            - For public samples (train/valid/test set): '<bbox_idx>-<action_idx>-<person_uid>-<frame_idx>-<clazz>-<id>-<UAI>-<image_size>.png'
            - For anonymous samples (challenge set): '<bbox_idx>-<action_idx>-<image_size>.png'
            The "id" field is the identifier of the player within an action. When the id is given as a number, it refers
             to the player jersey number. The jersey number is provided for a player if it can be seen at least once
             within one frame of the action. If the jersey number is not visible in any frame of the action, then this
             identifier is given as a letter.
        """
        info = {}
        splits = filename.split(".")[0].split("-")
        if len(splits) == 8:
            info["bbox_idx"] = int(splits[0])
            info["action_idx"] = int(splits[1])
            info["person_uid"] = splits[2]
            info["frame_idx"] = int(splits[3])
            info["clazz"] = splits[4]
            info["id"] = splits[5]
            info["UAI"] = splits[6]
            shape = splits[7].split("x")
            info["shape"] = (int(shape[0]), int(shape[1]))
        elif len(splits) == 3:
            info["bbox_idx"] = int(splits[0])
            info["action_idx"] = int(splits[1])
            shape = splits[2].split("x")
            info["shape"] = (int(shape[0]), int(shape[1]))
        else:
            raise ValueError(
                "Wrong sample filename format '{}'".format(filename))
        return info

    @staticmethod
    def get_bbox_index(filepath):
        return int(os.path.basename(filepath).split("-")[0])
