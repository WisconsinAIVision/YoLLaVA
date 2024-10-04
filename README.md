# 🌋👵🏻 Yo'LLaVA: Your Personalized LLaVA (NeurIPS 2024)

### [arXiv](https://arxiv.org/abs/2406.09400) | [BibTeX](#BibTeX) | [Project Page](https://thaoshibe.github.io/YoLLaVA/)


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

### Training

 🚧 Note: This code is under construction 🚧 -- While the base code is available, I have NOT tested the code and optimize the code yet -- Please check back for updates!

```
python train-multi-token.py --sks_name bo \
            --exp_name final5 --prefix_token 16 --epoch 15 \
            --model_path ./llava_ckpts/llava_ckpt \
            --data_root ./yollava-data/train \
            --user_prompt --recog_only --text_only --random_image

```

or run `bash bash.sh`

### Yo'LLaVA Dataset

To download the dataset, please intall Git Large File Storage (LFS) and clone the repository.
The dataset is in [`yollava-data`](https://github.com/WisconsinAIVision/YoLLaVA/tree/main/yollava-data) folder
```
    git lfs install
    git clone https://github.com/WisconsinAIVision/YoLLaVA.git

```

### 📝 TODO

- [ ] Optimization Code
    + [ ] Example training data (bo & mam)
- [ ] Pretrained for concepts
- [x] Dataset

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

Thank you (.❛ ᴗ ❛.)!