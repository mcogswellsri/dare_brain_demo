import requests
from itertools import product
from tqdm import tqdm



def main():
    '''
    The flask server is set up to permanently cache certain urls on disk
    when they are request. To fill the cache just request all these urls.
    '''
    base_url = 'http://dare-dev:5010'

    current_ids = list(range(0, 14))
    slice_ids = list(range(0, 155))[40:80]
    modalities = ['t1', 't2', 't2flair', 't1ce']
    tumor_types = ['ET', 'TC', 'WT']
    segment_types = ['gt', 'pred', 'counter']
    counter_types = ['image'] #, 'stats']
    norm_types = ['normal'] #, 'pixelwise']
    target_regions = ['entire'] #, 'fp', 'fn']
    layers = ['l1_early', 'l2_early', 'l3_early', 'l4_early',
              'l1_late', 'l2_late', 'l3_late', 'l4_late']

    print("@app.route('/slices/<int:current_id>/<path:slice_id>/<modality>')")
    for cid, sid, mod in tqdm(list(product(current_ids,
                                           slice_ids,
                                           modalities))):
        requests.get(base_url + f'/slices/{cid}/{sid}/{mod}')

    print("@app.route('/thumbnails/<int:current_id>')")
    for cid in tqdm(list(product(current_ids))):
        requests.get(base_url + f'/thumbnails/{cid}')

    print("@app.route('/segment/<source>/<tumor_type>/<int:entry_id>/<path:slice_id>/<modality>')")
    for tup in tqdm(list(product(segment_types,
                                 tumor_types,
                                 current_ids,
                                 slice_ids,
                                 modalities))):
        st, tt, cid, sid, mod = tup
        requests.get(base_url + f'/segment/{st}/{tt}/{cid}/{sid}/{mod}')

    print("@app.route('/counterfactual/<ctype>/<tumor_type>/<int:entry_id>/<path:slice_id>/<modality>')")
    for tup in tqdm(list(product(counter_types,
                                 tumor_types,
                                 current_ids,
                                 slice_ids,
                                 modalities))):
        ct, tt, cid, sid, mod = tup
        requests.get(base_url + f'/counterfactual/{ct}/{tt}/{cid}/{sid}/{mod}')

    print("@app.route('/counter_slices/<int:current_id>/<path:slice_id>/<modality>/<tumor_type>')")
    for tup in tqdm(list(product(current_ids,
                                 slice_ids,
                                 modalities,
                                 tumor_types))):
        cid, sid, mod, tt = tup
        requests.get(base_url + f'/counter_slices/{cid}/{sid}/{mod}/{tt}')

    print("@app.route('/counter_boxes/<int:current_id>/<path:slice_id>/<modality>/<tumor_type>')")
    for tup in tqdm(list(product(current_ids,
                                 slice_ids,
                                 modalities,
                                 tumor_types))):
        cid, sid, mod, tt = tup
        requests.get(base_url + f'/counter_boxes/{cid}/{sid}/{mod}/{tt}')

    print("@app.route('/gradcam/<int:current_id>/<path:slice_id>/<modality>/<target_cls>/<target_region>/<layer>/<norm_type>')")
    for tup in tqdm(list(product(current_ids,
                                 slice_ids,
                                 modalities,
                                 tumor_types,
                                 target_regions,
                                 layers,
                                 norm_types))):
        cid, sid, mod, tt, tr, lay, nt = tup
        requests.get(base_url + f'/gradcam/{cid}/{sid}/{mod}/{tt}/{tr}/{lay}/{nt}')


if __name__ == '__main__':
    main()
