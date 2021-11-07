# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import sys
import yaml


if __name__ == '__main__':
    with open('config.yaml') as f:
        args = yaml.safe_load(f)
    print(args)

    if len(sys.argv) < 2:
        print('Please set mode of HandyRL.')
        exit(1)

    mode = sys.argv[1]

    if mode == '--train' or mode == '-t':
        from handyrl.train import train_main as main
        main(args)
    elif mode == '--eval' or mode == '-e':
        from handyrl.evaluation import eval_main as main
        main(args, sys.argv[2:])
    else:
        print('Not found mode %s.' % mode)