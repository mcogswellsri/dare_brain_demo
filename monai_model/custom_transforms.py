import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import Randomizable, Transform, MapTransform


class RandGenerateRegion(Randomizable, Transform):
    """
    Given an image's segmentation, pick a class from the segmentation and a
    point labeled as that class. Return a volume centered around that point
    and represented as a mask tensor.
    """

    def __init__(self):
        self.side_len_min = 10
        self.side_len_max = 30

    def randomize(self, label: np.ndarray):
        # select the size of the box
        self.side_len = self.R.randint(low=self.side_len_min,
                                       high=self.side_len_max)
        # avoids an off by 1 error where the // in the next two lines
        # would make this side_len incaccurate 
        self.side_len = 2 * (self.side_len // 2) + 1

        # select a random category with at least some mass
        cls_choices = []
        label_mass = label.sum(axis=(1,2,3))
        for i in range(label_mass.shape[0]):
            if label_mass[i] > 0:
                cls_choices.append(i)
        assert len(cls_choices) > 0
        cls_idx = self.R.randint(len(cls_choices))
        self.cls = cls_choices[cls_idx]

        # select a voxel from that category for the center of the region
        points = np.where(label[self.cls])
        num_points = points[0].shape[0]
        point_idx = self.R.randint(num_points)
        self.center = np.array([dim[point_idx] for dim in points])

    def __call__(self, label: np.ndarray):
        """
        Args:
            label: Segmentation tensor of shape (n_labels, H[, W, ...])
                Each channel should have a 1 where the corresponding label is
                correct.
        """
        # sample a random volume
        self.randomize(label)

        # find the corners of that volume
        max_dims = label.shape[1:]
        corner_1 = np.maximum(0, self.center - self.side_len // 2)
        corner_2 = np.minimum(max_dims, self.center + self.side_len // 2)

        # convert the corners into a mask
        mask = np.zeros_like(label[0])[None] # (1HWD)
        mask[:,
             corner_1[0]:corner_2[0],
	     corner_1[1]:corner_2[1],
	     corner_1[2]:corner_2[2]] = 1
        return mask


class RandGenerateRegiond(Randomizable, MapTransform):
    """
    Dictionary-based version of RandGenerateRegion.

    Args:
        keys: The name of the segmentation label tensors.
        names: The name under which to store the generated masks.
    """

    def __init__(self, keys: KeysCollection, names: KeysCollection):
        super().__init__(keys)
        self.names = names
        self.rand_generate_region = RandGenerateRegion()

    def randomize(self, label: np.ndarray):
        self.rand_generate_region.randomize(label)

    def __call__(self, data):
        d = dict(data)
        for k, n in zip(self.keys, self.names):
            d[n] = self.rand_generate_region(d[k])
        return d


# from the monai examples: 
# https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/brats_segmentation_3d.ipynb
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes.
    See comments for details.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            # Task 1(ET) - ch 0 - ET - label 4
            # ET = l4
            ET = (d[key] == 4)
            result.append(ET)
            # Task 2(TC) - ch 1 - ET + NCR/NET - label 4 + label 1
            # TC = l4 + l1 = ET + l1
            NCR_NET = (d[key] == 1)
            TC = np.logical_or(ET, NCR_NET)
            result.append(TC)
            # Task 3(WT) - ch 2 - ED - label 4 + label 1 + label 2
            # WT = l4 + l1 + l2 = TC + l2
            ED = (d[key] == 2)
            WT = np.logical_or(TC, ED)
            result.append(WT)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def brats_label_to_raw(label, onehot=False):
    assert set(np.unique(label)) == {0., 1.}
    assert len(label.shape) == 4
    ET = label[0]
    TC = label[1]
    WT = label[2]
    #print(ET.sum(), TC.sum(), WT.sum())
    # l2 = WT - TC
    l2 = np.logical_and(WT, np.logical_not(TC))
    # l1 = TC - ET
    l1 = np.logical_and(TC, np.logical_not(ET))
    l4 = ET
    result = np.zeros(label.shape[1:])
    result[l1 == 1] = 1
    result[l2 == 1] = 2
    result[l4 == 1] = 4
    if onehot:
        return np.stack([l1, l2, l4])
    return result
