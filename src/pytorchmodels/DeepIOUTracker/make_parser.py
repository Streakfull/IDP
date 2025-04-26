import argparse


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str,
                        default=None, help="model name")

    parser.add_argument(
        "--path", default="../demo.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None,
                        type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize", default=None,
                        type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float,
                        default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float,
                        help="lowest detection threshold valid for tracks")
    # parser.add_argument("--track_low_thresh", default=0.5, type=float,
    #                     help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7,
                        type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float,
                        default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float,
                        default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float,
                        default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid",
                        default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.3,
                        help='threshold for rejecting low appearance similarity reid matches')
    return parser
