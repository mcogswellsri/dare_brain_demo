import shutil
import os.path
import json
import jinja2

import torch

def gradcam(model, batch, layers, target_f, eps=1e-8, neg=False, alphas=False):
    result = dict(batch)
    features, hook_handles = _save_features(model, layers)
    
    # forward model
    logits = model(batch['image'])
    result['preds'] = torch.sigmoid(logits)
    result['logits'] = logits

    # compute gradients
    loss, extra = target_f(model, logits, batch)
    loss.backward(retain_graph=True)
    
    # compute gradcam
    for lkey, _ in layers:
        fmap = features[lkey]
        if fmap.grad is None:
            raise Exception(f'Layer {lkey} has no gradient and it should. '
                            'Something is wrong.')
        alpha = fmap.grad.mean(dim=(2,3), keepdims=True)
        if alphas == 'pixelwise': # GradCAM
            vis = fmap.grad
            vis = vis - vis.min(dim=1, keepdims=True)[0]
            Z = vis.sum(dim=1, keepdims=True)
            vis = vis / Z
            result[lkey] = vis
        elif alphas: # just alphas
            eps = 1e-12
            alpha = alpha - alpha.min(dim=1, keepdims=True)[0]
            alpha = alpha / (alpha.max(dim=1, keepdims=True)[0] + eps)
            #alpha = alpha / (alpha + eps).sum(dim=1, keepdims=True)
            #assert ((alpha.sum(dim=1) - 1).abs() < 1e-4).all()
            #print(alpha[:, :, 0, 0, 0])
            result[lkey] = alpha.expand_as(fmap)
        else: # GradCAM
            gc = (fmap * alpha).sum(dim=1)
            if neg:
                gc = -gc
            gc = gc.relu()
            gc /= (gc.max() + eps)
            result[lkey] = gc
        hook_handles[lkey].remove()

    # copy and move to cpu/numpy
    for k in result:
        if torch.is_tensor(result[k]):
            result[k] = result[k].cpu().clone().detach().numpy()
    for k in extra:
        assert k not in result
        if torch.is_tensor(extra[k]):
            result[k] = extra[k].cpu().clone().detach().numpy()
        else:
            result[k] = extra[k]
    return result


def _save_features(model, layers):
    features = {}
    hook_handles = {}
    named_modules = dict(model.named_modules())
    for key, _ in layers:
        mod = named_modules[key]
        #print(f'saving {key}')
        def save_output(module, input, output, name=key):
            features[name] = output
            output.requires_grad_()
            output.retain_grad()
        def save_input(module, input, name=key):
            assert len(input) == 1, 'only single argument modules supported'
            input = input[0]
            features[name] = input
            input.requires_grad_()
            input.retain_grad()
        if key == '':
            handle = mod.register_forward_pre_hook(save_input)
        else:
            handle = mod.register_forward_hook(save_output)
        hook_handles[key] = handle
    return features, hook_handles


_label_space = 'normal' # ET, TC, WT
#_label_space = 'raw' # background, NCR/NET, ED, ET
def set_label_space(space):
    global _label_space
    assert space in ['normal', 'raw']
    _label_space = space

_agg_dims = (1, 2, 3) # for 3d unet
#_agg_dims = (1, 2) # for 2d unet
def set_agg_dims(agg_dims):
    global _agg_dims
    _agg_dims = agg_dims


def _scores_from_logits(logits, label_ch):
    if _label_space == 'normal':
        scores = logits[:, label_ch]
    elif _label_space == 'raw':
        # label_ch in [0, 1, 2] corresponding to [ET, TC, WT]
        # logits[0, :] has background, NCR/NET, ED, ET
        # 'NCR/NET' , 'ED' , 'ET'                                                
        # l1, l2, l4                                                             
        logits = logits[: 1:] 
        if label_ch == 0: # ET = 'ET' = l4 = index 2
            scores = logits[:, 2]
        elif label_ch == 1: # TC = l1 + ET = l1 + l4 = index 0 + index 2
            scores = (logits[:, 0] + logits[:, 2]) / 2
        elif label_ch == 2: # WT = l2 + TC = l2 + l1 + l4 = index 1 + index 0 + index 2
            scores = (logits[:, 1] + logits[:, 0] + logits[:, 2]) / 3
    return scores

def target_entire_cls(label_ch):
    def f_(model, logits, batch):
        scores = _scores_from_logits(logits, label_ch)
        return scores.mean(dim=_agg_dims).sum(dim=0), {} # spatial mean, batch sum
    return f_


def target_partial(error_type, label_ch):
    def f_(model, logits, batch):
        extra = {}
        gt = batch['seg']
        pred = batch['y_pred']
        assert set(gt.unique().tolist()) <= set([0., 1.])
        assert set(pred.unique().tolist()) <= set([0., 1.])
        if error_type == 'tp':
            mask = pred * gt
        elif error_type == 'fp':
            mask = pred * (1 - gt)
        elif error_type == 'fn':
            mask = (1 - pred) * gt
        elif error_type == 'tn':
            mask = (1 - pred) * (1 - gt)
        elif error_type == 'uncertain':
            probs = batch['y_pred_probs']
            mask = (0.07 < probs) & (probs < 0.93)
            mask = mask.to(torch.float)
        target = (logits * mask)[:, label_ch].mean(dim=(1,2)).sum(dim=0)
        extra = {
            'error_type': error_type,
            'error_mask': mask,
        }
        return target, extra
    return f_


def main(args):
    '''
    Generate an html file that displays inputs/output/visualizations
    from pre-computed (use eval.py) images.
    '''
    meta_file = os.path.join(args.model_dir, args.result_dname, 'vis_meta.json')
    with open(meta_file) as f:
        vis_meta = json.load(f)

    with open('vis_template.html') as f:
        template = jinja2.Template(f.read())

    shutil.copy('unet_annotated_diagram.png', os.path.join(args.model_dir, args.result_dname, 'unet_annotated_diagram.png'))

    out_fname = os.path.join(args.model_dir, args.result_dname, 'vis_examples.html')
    with open(out_fname, 'w') as f:
        #print(out_fname)
        f.write(template.render(vis_meta))


if __name__ == '__main__':
    from options import parser
    args = parser.parse_args()
    main(args)
