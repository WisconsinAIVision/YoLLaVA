# 🌋👵🏻 Yo'LLaVA: Your Personalized LLaVA (NeurIPS 2024)

### [arXiv](https://arxiv.org/abs/2406.09400) | [BibTeX](#BibTeX) | [Project Page](https://thaoshibe.github.io/YoLLaVA/) | [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/93737.png?t=1729115312.34047)


<!-- Yo'LLaVA <img src='./images/yollava.png' width=150> is LLaVA <img src='./images/llava_logo.png' width=150>, but can provide personlized conversation! -->

<!-- <p>
    Yo'LLaVA <img src="./images/yollava.png" width="150" align="middle"> is LLaVA <img src="./images/llava_logo.png" width="150" align="middle">, but can provide personalized conversation!
</p> -->

<!-- <table style="width: 100%; text-align: center;">
    <tr>
        <td style="text-align: center; vertical-align: middle;">
            <a href="https://llava-vl.github.io/"> 🌋 LLaVA</a> <br>
            <img src="./images/llava_logo.png" width="150" alt="LLaVA Logo">
        </td>
        <td style="text-align: center; vertical-align: middle;">
            + 👵🏻 𝓹𝓮𝓻𝓼𝓸𝓷𝓪𝓵𝓲𝔃𝓮𝓭 abilities ✨
        </td>
        <td style="text-align: center; vertical-align: middle;">
            =
        </td>
        <td style="text-align: center; vertical-align: middle;">
            <a href='https://thaoshibe.github.io/YoLLaVA'>🌋 <b>Yo</b>ur <b>LLaVA</b> 👵🏻</a><br>
            <img src="./images/yollava.png" width="150" alt="YoLLaVA Image">
        </td>
    </tr>
</table> -->

<img src="./images/yollava-without-background.png" width="300" alt="YoLLaVA Image">

<!-- | ![./images/yollava-without-background.png](./images/yollava-without-background.png) | -->


☆.。.:*・°☆.。.:*・°

[🌋👵🏻 **Yo'LLaVA: Your Personalized Language and Vision Assistant**](https://thaoshibe.github.io/YoLLaVA/) (NeurIPS 2024)<br>
[Thao Nguyen ✨](https://thaoshibe.github.io/), [Haotian Liu](https://hliu.cc/), [Mu Cai](https://pages.cs.wisc.edu/~mucai/), [Yuheng Li](https://yuheng-li.github.io/), [Utkarsh Ojha](https://utkarshojha.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) <br>
🦡 University of Wisconsin-Madison

| ![./assets/images/teaser.png](./images/github-teaser.png) |
|:--:|
| Given just a few images of a novel subject (e.g., a dog named `<bo>`, a person named `<thao>`), Yo’LLaVA learns to facilitate textual/visual conversations centered around that subject. |

☆.。.:*・°☆.。.:*・°

> **Abstract**: Large Multimodal Models (LMMs) have shown remarkable capabilities across a variety of tasks (e.g., image captioning, visual question answering). While broad, their knowledge remains generic (e.g., recognizing a dog), and they are unable to handle personalized subjects (e.g., recognizing a user's pet dog). Human reasoning, in contrast, typically operates within the context of specific subjects in our surroundings. For example, one might ask, "What should I buy for my dog's birthday?"; as opposed to a generic inquiry about "What should I buy for a dog's birthday?". Similarly, when looking at a friend's image, the interest lies in seeing their activities (e.g., "my friend is holding a cat"), rather than merely observing generic human actions (e.g., "a man is holding a cat"). In this paper, we introduce the novel task of personalizing LMMs, so that they can have conversations about a specific subject. We propose Yo'LLaVA, which learns to embed a personalized subject into a set of latent tokens given a handful of example images of the subject. Our qualitative and quantitative analyses reveal that Yo'LLaVA can learn the concept more efficiently using fewer tokens and more effectively encode the visual attributes compared to strong prompting baselines (e.g., LLaVA).

### Training/ Testing

**Installation**: This code is directly built on top of [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install). Please follow [LLaVA's installation](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install)!

 🚧 Note: This code is under construction 🚧 -- While the base code is available, I have NOT tested the code and optimize the code yet -- Please check back for updates!

```
python train-multi-token.py --name bo \
            --exp_name final5 --prefix_token 16 --epoch 15 \
            --model_path ./llava_ckpts/llava_ckpt \
            --data_root ./yollava-data/train \
            --user_prompt --recog_only --text_only --random_image
```

or run `bash train.sh`

To test, plesase refer to `test-sks.py` and `test-sks-qa.py`.

### Yo'LLaVA Dataset

<img src="./images/yollava-dataset.png" width="300" alt="YoLLaVA Data">

To download the dataset, please intall Git Large File Storage (LFS) and clone the repository.
The dataset is in [`yollava-data`](https://github.com/WisconsinAIVision/YoLLaVA/tree/main/yollava-data) folder
```
git lfs install
git clone https://github.com/WisconsinAIVision/YoLLaVA.git
```

The simple Visual Question Answering json file is located in `yollava-visual-qa.json` with the following format:
```
{
    "./yollava-data/test/bo/0.png":
    {
        "question": "What is the primary color of <sks>'s fur?",
        "options":
        {
            "A": "Brown",
            "B": "Grey"
        },
        "correct_answer": "A"
    }
}
```


##### Retrieved Negative Examples

For your convenience, retrieved negative examples are provided in this [Google Drive](https://drive.google.com/drive/folders/1bqM5y0-Kw26R5T4kfaeUAZzeKqIdREdU?usp=sharing).

Please not that these images are retrieved from [LAION-2B with CLIP](https://github.com/rom1504/clip-retrieval/tree/main); and we do **NOT** own the rights to these images, and these images are **purely for research purposes**.

<img src="./images/negative-example.png" width="300" alt="Example of negative retrieved">

Please download the `yollava-data.zip` in [Google Drive](https://drive.google.com/drive/folders/1bqM5y0-Kw26R5T4kfaeUAZzeKqIdREdU?usp=sharing) and unzip it.
In the folder, you can also find the json file with CLIP similarity scores. Folder structure:

```
yollava-data
├── train
│   ├── bo
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
│   │   └── negative_example
│   │       ├── 76618997f6ce14d73ccde567a6c8eabb.png
│   │       ├── eca8f558d3c4423351f45e87fb8ee5f9.png
│   │       ├── ...
│   │       └── scores.json
├── test
│   ├── bo
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
```

The json file has the following format:
```
{
    "image": "51df89957cb840afa91b37db9669fd1b",
    "image_path": "/mnt/localssd/code/data/yollava-data/train/bo/negative_example/51df89957cb840afa91b37db9669fd1b.png",
    "clip_score": 0.6376656889915466
}
```

##### Some pretrained concepts

We also provide some pretrained concepts in this [Google Drive](https://drive.google.com/drive/folders/1bqM5y0-Kw26R5T4kfaeUAZzeKqIdREdU?usp=sharing). This pretrained concepts library includes:

| <img src="./yollava-data/train/bo/1.png" height="100"><br>`<bo>` | <img src="./yollava-data/train/butin/0.png" height="100"><br>`<butin>` | <img src="./yollava-data/train/ciin/0.png" height="100"><br>`<ciin>` | <img src="./yollava-data/train/denisdang/0.png" height="100"><br>`<denisdang>` | <img src="./yollava-data/train/dug/0.png" height="100"><br>`<dug>` |
| --- | --- | --- | --- | --- |
| <img src="./yollava-data/train/khanhvy/0.png" height="100"><br>`<khanhvy>` | <img src="./yollava-data/train/mam/0.png" height="100"><br>`<mam>` | <img src="./yollava-data/train/marie-cat/0.png" height="100"><br>`<marie-cat>` | <img src="./yollava-data/train/oong/1.png" height="100"><br>`<oong>` | <img src="./yollava-data/train/phuc-map/0.png" height="100"><br>`<phuc-map>` |
| <img src="./yollava-data/train/thao/3.png" height="100"><br>`<thao>` | <img src="./yollava-data/train/thuytien/0.png" height="100"><br>`<thuytien>` | <img src="./yollava-data/train/viruss/0.png" height="100"><br>`<viruss>` | <img src="./yollava-data/train/willinvietnam/0.png" height="100"><br>`<willinvietnam>` | <img src="./yollava-data/train/yuheng/2.png" height="100"><br>`<yuheng>` |




The `best.pt` is the checkpoint that have higest recognition accuracy in the train set. Other checkpoints are also provided in the folder.

### BibTeX

```
@misc{nguyen2024yollavapersonalizedlanguagevision,
      title={Yo'LLaVA: Your Personalized Language and Vision Assistant}, 
      author={Thao Nguyen and Haotian Liu and Yuheng Li and Mu Cai and Utkarsh Ojha and Yong Jae Lee},
      year={2024},
      eprint={2406.09400},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.09400}, 
}
```

### Acknowledgement

This code is heavily borrowed from:
- Awesome [LLaVA](https://github.com/haotian-liu/LLaVA)!
- [Textual Inversion](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py)

Thank you (.❛ ᴗ ❛.)!
