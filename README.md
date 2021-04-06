# f-POVI

<p align="center">
<img src="http://ml.cs.tsinghua.edu.cn/~ziyu/static/fpovi/i.gif" width="500">
</p>

Code for *Function Space Particle Optimization for Bayesian Neural Networks*, ICLR 2019.

Please reach out at <wzy196@gmail.com> for questions.

*For an imperative implementation, check out the [tf2 branch](https://github.com/thu-ml/fpovi/tree/tf2).*

## Dependencies

- requirements.txt
- thu-ml/zhusuan (@`1011dd9`)
- meta-inf/experiments (@`60c0a77`)

## Usage

See scripts in `exps/`.

You are recommended to look at `ffn/` if you want a minimal implementation to build upon.

## Acknowledgements

This repository contains code adapted from other sources:

- The `dcb` directory is a fork of the [deep contextual bandit library](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits) by Riquelme et al.
- The convnet implementation is adapted from [Tensorpack](https://github.com/tensorpack/tensorpack);
- The adversarial attack code (for feed-forward network) is taken from Yingzhen Li's [code for alpha-dropout](https://github.com/YingzhenLi/Dropout_BBalpha/blob/master/attacks_tf.py);
- The UCI data processing script is taken from the [doubly stochasitc DGP code](https://github.com/ICL-SML/Doubly-Stochastic-DGP), from Salimbeni et al.

## Citation

```
@inproceedings{
    wang2018function,
    title={Function Space Particle Optimization for {B}ayesian Neural Networks},
    author={Ziyu Wang and Tongzheng Ren and Jun Zhu and Bo Zhang},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=BkgtDsCcKQ},
}
```
