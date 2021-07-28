Overview
===

In treating gliomas, a type of brain cancer, radiologists need to localize
tumor tissue extent in magnetic resonance images. This is laborious and
potentially marred with uncertainties as even amongst expert neuro-radiologist
there may be disagreement about the exact glioma parameters. Automatic brain
tumor segmentation in MR images aims to facilitate this process. Glioma
segmentation accuracy has improved greatly in recent years, but failures still
may occur. We aim to help radiologists better understand these failures so they
can more effectively use automatic segmentation approaches.

This demo builds on prior work that has shown attentional and counterfactual
explanations help users understand machine learning models. It implements an
interface, which allows users to learn about segmentation models through
interaction. We apply the ideas of attentional and counterfactual
explanations by allowing users to ask questions like "Which regions of the
brain were you looking at?" and "How would the model behave in certain similar
but different cases?" Specifically, we use heatmaps from GradCAM to answer the
first question and counterfactual segmentations generated from inpainted
version of original brain images to answer the second question. The rest of
this README explains how the code is structured and how to run the interface on
your own machine.

Running this Demo
===

Setup
---

1. Clone the code for the inpainting GAN into the root directory of
this repository.

```
$ git clone https://github.com/JiahuiYu/generative_inpainting
```

2. Copy the inpainting configuration file.

```
$ cp inpaint_config_files/inpaint.yml generative_inpainting/
```

Starting the server
---

To start the demo, run

    $ doit.sh

This compiles and runs a docker container the code from this directory placed
in `/usr/src/app/` and the data under `data/` mounted according
to `local_config.json`. The container then runs `python brats_demo_server.py`,
which is the main driver for the demo. This serves `brats_demo.html` to
a url like `http://localhost:5007/brats_demo/` then waits for interactive
requests from the webpage. Depending on the request, the result may be served
from the cache mounted from `data/brats_cache/` or it may dynamically computed
the requested visualization. This can be expensive and slow (e.g., for
generating counterfactuals) or it can be be quicker (e.g., for gradcams and
tumor segments).

Requirements
---

To run start the docker container using the provided script our environment
included python3.7 with jinja2 and pyyaml installed.

Configuration
---

The port the server is listens on is configured in two places. To change it
modify the `EXPOSE` line in `Dockerfile` and modify the default `--port`
argument at the top of `brats_demo_server.py`.

The server is configured to cache most of its responses that may take a while
to computein the `/usr/src/app/brats_cache/` directory. Initially a small subset
of the data available in the application is cached, but as it is used more of
the cache is filled. The cache can also be filled in advance by running
`cache_everything.py`. This takes a while, so adjust the parameters at the top
to target a subset of the data.

Code Structure
---

Visualizations are dynamically generated on the backend via calls to functions
in `brats_demo_server.py`. The following functions are good places to start.
They're wrappers that call other things to eventually produce the desired
functionality.

* `send_slice(...)`: Sends the slice of brain specified in the url.
* `send_segment(...)`: Send a segment of particular type, whether that be
    ground truth, the model's prediction, or the counterfactual segmentation
    for the selected slice.
* `send_gradcam(...)`: Send the gradcam requested in the url, computing it on
    the fly if necessary.
* `send_counterfactual_slice(...)`: Compute the inpainted image that inpaints
    a box around the specified tumor and send it back.

