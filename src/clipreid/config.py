from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    # automatically load pretrained model weights if available
    cfg.model.pretrained = True
    cfg.model.load_weights = ''  # path to model weights
    cfg.model.resume = ''  # path to checkpoint for resume training

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'raw_dataset'
    cfg.data.sources = ['soccernetv3']
    # choose from ['soccernetv3', 'soccernetv3_test', 'soccernetv3_challenge']
    cfg.data.targets = ['soccernetv3']
    # metric for evaluation, choose from 'default', 'cuhk03', 'soccernetv3'
    cfg.data.eval_metric = 'soccernetv3'
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    # cfg.data.height = 128  # image height
    # cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip']  # data augmentation
    cfg.data.k_tfm = 1  # number of times to apply augmentation to an image independently
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log
    cfg.data.load_train_targets = False  # load training set from target dataset

    # specific datasets
    cfg.market1501 = CN()
    # add 500k distractors to the gallery set for market1501
    cfg.market1501.use_500k_distractors = False
    cfg.cuhk03 = CN()
    # use labeled images, if False, use detected images
    cfg.cuhk03.labeled_images = False
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.soccernetv3 = CN()
    # Use 'training_subset'% of total number of training set actions at training
    cfg.soccernetv3.training_subset = 1.0
    # stage. Use this option for faster training. Set to 1.0 to use full training set.

    # sampler
    cfg.sampler = CN()
    # sampler for source train loader
    cfg.sampler.train_sampler = 'RandomIdentitySampler'
    # sampler for target train loader
    cfg.sampler.train_sampler_t = 'RandomIdentitySampler'
    # number of instances per identity for RandomIdentitySampler
    cfg.sampler.num_instances = 6
    # number of cameras to sample in a batch (for RandomDomainSampler)
    cfg.sampler.num_cams = 10
    # number of datasets to sample in a batch (for RandomDatasetSampler)
    cfg.sampler.num_datasets = 1

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15  # number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly'  # how to sample images from a tracklet
    cfg.video.pooling_method = 'avg'  # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 96
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = [
        'classifier'
    ]  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ['classifier']  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed
    cfg.train.warmup_lr = 0.0  # for training transformers
    cfg.train.warmup_steps = 0
    cfg.train.min_lr = 1e-6  # for training transformers
    cfg.train.warmup_epochs = 0  # for training transformers

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss
    cfg.loss.triplet.weight_tc = 0.  # weight to balance centroid triplet loss
    cfg.loss.triplet.weight_cc = 0.  # weight to balance loss from distance between centroids
    cfg.loss.triplet.topk = 1  # Mine Top k hard negatives
    cfg.loss.triplet.bottomk = 1  # Mine bottom hard positives

    # test
    cfg.test = CN()
    cfg.test.batch_size = 96
    cfg.test.dist_metric = 'cosine'  # distance metric, ['euclidean', 'cosine']
    # normalize feature vectors before computing distance
    cfg.test.normalize_feature = False
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    # evaluation frequency (-1 means to only test after training)
    cfg.test.eval_freq = -1
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank = False
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    # export query to gallery ranking results to JSON file in 'data.save_dir' for each
    cfg.test.export_ranking_results = False
    # target dataset. To be used for external evaluation and submission on EvalAI
    cfg.use_gpu = True
    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'k_tfm': cfg.data.k_tfm,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'load_train_targets': cfg.data.load_train_targets,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'num_cams': cfg.sampler.num_cams,
        'num_datasets': cfg.sampler.num_datasets,
        'train_sampler': cfg.sampler.train_sampler,
        'train_sampler_t': cfg.sampler.train_sampler_t,
        # image dataset specific
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'soccernetv3_training_subset': cfg.soccernetv3.training_subset,
    }
