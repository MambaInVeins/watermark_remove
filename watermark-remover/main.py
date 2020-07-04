import pathlib, argparse, os, time, datetime, logging, coloredlogs
from autowatermarkremoval.solver import solve
from autowatermarkremoval.remover import remove
coloredlogs.install(level='DEBUG')
log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='\n            Watermark Remover v0.2\n            https://github.com/whitelok/watermark-remover\n            ')
    # --mode 两种模式 train模式和remove模式
    parser.add_argument('--mode', type=str, required=True, help='\n        train: train and get a watermark and a alpha image\n        remove: just remove watermark from a image\n        ')
    # --dataset train模式下用于训练水印
    parser.add_argument('--dataset', type=str, help='\n        images path for training a watermark, just for train mode\n        ')
    # --watermark_path remove模式水印位置
    parser.add_argument('--watermark_path', type=str, help='\n        load the watermark image, just for remove mode\n            ')
    # --alpha_path remove模式Alpha图片
    parser.add_argument('--alpha_path', type=str, help='\n        load the alpha image, just for remove mode\n        ')
    # --image_path remove模式原图位置
    parser.add_argument('--image_path', type=str, help='\n        load the image you want to remove the watermark, just for remove mode\n        ')
    # --iters train模式设置训练迭代时间
    parser.add_argument('--iters', type=int, default=3, help='\n        set training iteration times\n        ')
    # --watermark_threshold 将水印阈值设置在（0到1）之间
    parser.add_argument('--watermark_threshold', default=0.7, type=float, help='\n        set watermark threshold between (0 to 1)\n        ')
    # --save_result 是否将结果另存为图像
    parser.add_argument('--save_result', type=bool, default=True, help='\n        whether save result as images\n        ')
    options = parser.parse_args()
    # 设置试用日期
    date_time = datetime.datetime.strptime('2020-07-30', '%Y-%m-%d')
    if datetime.datetime.now() < date_time:
        if 'train' == options.mode:
            if not options.dataset is not None:
                raise AssertionError('need a --dataset for training.')
            else:
                srcpath = options.dataset
                destpath = os.path.join(options.dataset, 'result')
                # 判断是否是绝对路径 
                if not os.path.isabs(srcpath):
                    srcpath = os.path.join(os.path.curdir, srcpath)
                if not os.path.exists(destpath):
                    os.mkdir(destpath)
            solve(pathlib.Path(srcpath), pathlib.Path(destpath), options.iters, options.watermark_threshold, options.save_result)
        else:
            if 'remove' == options.mode:
                assert options.watermark_path and options.alpha_path and options.image_path, 'need --watermark_path and --alpha_path and --image_path'
                remove(pathlib.Path(options.watermark_path), pathlib.Path(options.alpha_path), pathlib.Path(options.image_path))
            else:
                log.error('Unknown mode !!!')
    else:
        print("Can't use on your computer, please ask author your license, sorry.")
        exit(0)
# okay decompiling main.pyc
