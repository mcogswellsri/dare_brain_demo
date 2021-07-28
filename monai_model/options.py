import argparse

parser = argparse.ArgumentParser()

parser.add_argument('model_dir')

parser.add_argument('--train-data',
                    default='./data/MICCAI_BraTS2020_TrainingData_train/')
parser.add_argument('--val-data',
                    default='./data/MICCAI_BraTS2020_TrainingData_dev/')
parser.add_argument('--train-epochs', default=300, type=int)

parser.add_argument('--checkpoint-epoch', type=int)
parser.add_argument('--split', default='val', choices=['val', 'all'])


# model params
parser.add_argument('--model', default='unet',
                    choices=['unet', 'local_unet'])
parser.add_argument('--up-mode', default='transpose',
                    choices=['transpose', 'resize'],
                    help='Which method to use for upsampling feature maps.\n'
                         '    tranpose - transposed convolution\n'
                         '    resize - resize then conv (avoids artifacts)')


# for debugging
parser.add_argument('--max-examples', type=int, default=None)


# for evaluation / visualization
parser.add_argument('--vis-page', type=int, default=0,
                    help='Generate a web page that shows many visualizations '
                         'for many images')
parser.add_argument('--vis-nifti', type=int, default=0,
                    help='Generate nifti volumes containing 3d attention '
                         'visualizations')
parser.add_argument('--contest-submission', type=int, default=0,
                    help='Generate segmentation outputs as nifti for contest '
                         'submission')
parser.add_argument('--survival-features', type=int, default=0,
                    help='Generate a file containing features for surviaval '
                         'prediction.')
parser.add_argument('--save-metrics', type=int, default=0,
                    help='Compute and save metrics')
parser.add_argument('--alpha-analysis', type=int, default=0,
                    help='Save GradCAM alphas for analysis')
parser.add_argument('--result-dname', default='results')
parser.add_argument('--vis-key', default=None)
parser.add_argument('--layer-key', default=None)

parser.add_argument('--pred-thresh', type=float, default=0.0)
